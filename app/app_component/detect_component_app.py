from PyQt5.QtWidgets import QWidget, QListWidgetItem, QListWidget
import time
from PyQt5 import QtCore
from enum import Enum
import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import sys
import csv
from itertools import islice
from PyQt5.QtGui import QImage, QPixmap, QIcon
from app.ui_component.detect_component import Ui_DetectComponent
from threading import Thread, Lock
from app.pipeline_module.core.base_module import *
from app.pipeline_module.core.task_solution import *
from app.pipeline_module.video_modules import *
from app.pipeline_module.yolo_modules import *
from app.pipeline_module.vis_modules import ObjectDetectVisModule

SOURCE_DIR_RELATIVE = "datasets/test_dataset"
yolov5_weight = './weights/yolov5s.torchscript.pt'
alphapose_weight = './weights/halpe136_mobile.torchscript.pth'
classroom_action_weight = './weights/classroom_action_lr_front_v2_sm.torchscript.pth'
device = 'cpu'


def second2str(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


class QListWidgetItemForVideo(QListWidgetItem):
    class SourceType(Enum):
        FILE = "file"
        CAMERA = "camera"

    def __init__(self, list_widget, name, src, type=SourceType.FILE.value):
        """

        :param list_widget:
        :param name:
        :param src:
        :param type: 'file'(default) or 'camera'
        """
        super(QListWidgetItemForVideo, self).__init__()
        self.list_widget: QListWidget = list_widget
        self.type = type
        # cap = cv.VideoCapture(src)
        # if not cap.isOpened():
        #     raise Exception()
        # ret, frame = cap.read()
        # cap.release()
        # icon = QIcon()
        # icon.addPixmap(QPixmap(frame), QIcon.Normal, QIcon.Off)
        # self.setIcon(icon)
        self.setText(name + "(" + src + ")")
        self.src = src

    def add_item(self):
        len = self.list_widget.count()
        for i in range(len):
            if self.list_widget.item(i).src == self.src:
                return
        self.list_widget.addItem(self)


# 可以建一个buffer队列，每次处理完几帧后通过信号等方式，通知label从队列里去取处理后的每帧数据进行渲染，这样卡顿会好一些

class OffsetList(list):
    def __init__(self, seq=()):
        super(OffsetList, self).__init__(seq)
        self.offset = 0

    def min_index(self):
        return self.offset

    def max_index(self):
        return self.offset + len(self) - 1

    def __getitem__(self, item):
        return super(OffsetList, self).__getitem__(max(0, item - self.offset))

    def append(self, __object) -> None:
        super(OffsetList, self).append(__object)

    def pop(self, **kwargs):
        self.offset += 1
        super(OffsetList, self).pop(0)

    def clear(self) -> None:
        self.offset = 0
        super(OffsetList, self).clear()


class DetectComponentApp(QWidget, Ui_DetectComponent):
    push_frame_signal = QtCore.pyqtSignal(DictData)

    def __init__(self, *args, **kwargs):
        super(DetectComponentApp, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.video_source = 0
        self.frame_data_list = OffsetList()
        self.opened_source = None
        self.playing = None
        self.playing_real_time = False
        self.num_of_passing = 0
        self.num_of_peep = 0
        self.num_of_gazing_around = 0

        # widget 初始化
        self.widget_init()

        # 视频播放锁
        self.open_source_lock = Lock()

        # 事件注册
        self.event_register()

    def event_register(self):
        """
        open_source -> 通过emit把module处理过后的数据发送到 push_frame
        push_frame 把图片存到内存(frame_data_list)， 然后设置进度条最大和最小值。
        进度条 从 -1 变为0 ，触发change_frame,change_frame 中循环触发change_frame

        :return:
        """

        self.open_source_btn.clicked.connect(
            lambda: self.open_source(self.video_now_source_path.text()
                                     if len(self.video_now_source_path.text()) != 0 else 0))

        self.video_resource_list.itemClicked.connect(lambda item: self.open_source(item.src))
        self.video_resource_file_list.itemClicked.connect(lambda item: self.open_source(item.src))

        self.close_source_btn.clicked.connect(self.close_source)
        self.play_video_btn.clicked.connect(self.play_video)
        self.stop_playing_btn.clicked.connect(self.stop_playing)
        self.video_process_bar.valueChanged.connect(self.change_frame)

        # 自定义信号
        self.push_frame_signal.connect(self.push_frame)

    def widget_init(self):
        """
        widget 初始化
        """
        self.init_video_file_source()  # 初始化视频文件源
        self.init_video_camera_source()  # 初始化摄像源

    def init_video_file_source(self):
        # 测试文件路径
        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[2]
        source_dir = (ROOT / SOURCE_DIR_RELATIVE).absolute()

        # 初始化文件夹
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)

        videos = [*filter(lambda x: x.endswith('.avi') or x.endswith('.mp4'), os.listdir(source_dir))]

        # 获取默认目录下的视频文件
        for video_name in videos:
            try:
                item = QListWidgetItemForVideo(self.video_resource_file_list,
                                               video_name,
                                               os.path.join(source_dir, video_name), )
            except Exception:
                print("没有找到视频文件:", video_name)
            else:
                item.add_item()

        # 获取csv文件中指定的视频文件
        video_file_list_csv = source_dir / 'video_sources.csv'
        if not os.path.exists(video_file_list_csv):
            with open(video_file_list_csv, 'w', encoding='utf-8') as f:
                f.write("")

        with open(video_file_list_csv, 'r', encoding='utf-8') as f:
            file_reader = csv.reader(f, delimiter=',')
            for row in islice(file_reader, 1, None):
                src: str = row[1]
                if src.startswith("."):
                    src = str((source_dir / src).absolute())
                QListWidgetItemForVideo(self.video_resource_file_list, row[0], src).add_item()

    def init_video_camera_source(self):
        # 添加视频通道
        QListWidgetItemForVideo(self.video_resource_list, "摄像头", "0",
                                type=QListWidgetItemForVideo.SourceType.CAMERA.value).add_item()

    def open_source(self, source):
        self.open_source_lock.acquire(blocking=True)
        if self.opened_source is not None:
            self.close_source()
        # Loading
        frame = np.zeros((480, 640, 3), np.uint8)
        (f_w, f_h), _ = cv2.getTextSize("Loading", cv2.FONT_HERSHEY_TRIPLEX, 1, 2)

        cv2.putText(frame, "Loading", (int((640 - f_w) / 2), int((480 - f_h) / 2)),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1, (255, 255, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.video_screen.width() - 9, self.video_screen.height() - 9))  # 调整图像大小
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                       image_width * image_depth,
                       QImage.Format_RGB888)
        self.video_screen.setPixmap(QPixmap.fromImage(frame))

        # 启动视频源
        def open_source_func(self):
            fps = 12
            # self.opened_source = TaskSolution() \
            #     .set_source_module(VideoModule(source, fps=fps)) \
            #     .set_next_module(YoloV5Module(yolov5_weight, device)) \
            #     .set_next_module(ObjectDetectModule()) \
            #     .set_next_module(ObjectDetectVisModule(lambda d: self.push_frame_signal.emit(d)))
            self.opened_source = TaskSolution() \
                .set_source_module(VideoModule(source, fps=fps)) \
                .set_next_module(ObjectDetectVisModule(lambda d: self.push_frame_signal.emit(d)))
            self.opened_source.start()
            self.playing_real_time = True
            self.open_source_lock.release()
            fps = 12

        Thread(target=open_source_func, args=[self]).start()

    def close_source(self):
        if self.opened_source is not None:
            self.stop_playing()
            self.opened_source.close()
            self.opened_source = None
            self.frame_data_list.clear()
            self.video_process_bar.setMaximum(-1)
            self.playing_real_time = False
            # self.cheating_list.clear()
            # self.real_time_catch_list.clear()

    def push_frame(self, data):
        try:
            max_index = self.frame_data_list.max_index()
            time_process = self.frame_data_list[max_index].time_process if len(self.frame_data_list) > 0 else 0
            data.time_process = time_process + data.interval
            # 添加帧到视频帧列表
            self.frame_data_list.append(data)
            while len(self.frame_data_list) > 500:
                self.frame_data_list.pop()
            self.video_process_bar.setMinimum(self.frame_data_list.min_index())
            self.video_process_bar.setMaximum(self.frame_data_list.max_index())

            # 添加到作弊列表
            # data.frame_num = max_index + 1
            # if data.num_of_cheating > 0 and self.check_cheating_change(data):
            #     self.add_cheating_list_signal.emit(data)

            # 判断是否进入实时播放状态
            if self.playing_real_time:
                self.video_process_bar.setValue(self.video_process_bar.maximum())

        except Exception as e:
            print(e)

    def playing_video(self):
        try:
            while self.playing is not None and not self.playing_real_time:
                current_frame = self.video_process_bar.value()
                max_frame = self.video_process_bar.maximum()
                if current_frame < 0:
                    continue
                elif current_frame < max_frame:
                    data = self.frame_data_list[current_frame]
                    if current_frame < max_frame:
                        self.video_process_bar.setValue(current_frame + 1)
                    time.sleep(data.interval)
                else:
                    self.stop_playing()
                    self.playing_real_time = True
        except Exception as e:
            print(e)

    def play_video(self):
        if self.playing is not None:
            return
        self.playing = Thread(target=self.playing_video, args=())
        self.playing.start()

    def change_frame(self):
        try:
            if len(self.frame_data_list) == 0:
                return
            current_frame = self.video_process_bar.value()
            max_frame = self.video_process_bar.maximum()
            self.playing_real_time = current_frame == max_frame  # 是否开启实时播放
            # 更新界面
            data = self.frame_data_list[current_frame]
            maxData = self.frame_data_list[max_frame]
            frame = data.frame_anno if self.show_box.isChecked() else data.frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.video_screen.width() - 9, self.video_screen.height() - 9))  # 调整图像大小
            image_height, image_width, image_depth = frame.shape
            frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                           image_width * image_depth,
                           QImage.Format_RGB888)
            self.video_screen.setPixmap(QPixmap.fromImage(frame))
            # 显示时间
            current_time_process = second2str(data.time_process)
            max_time_process = second2str(maxData.time_process)

            self.time_process_label.setText(f"{current_time_process}/{max_time_process}")
        except Exception as e:
            print(e)

    def stop_playing(self):
        if self.playing is not None:
            # thread 设置为None之后就就出循环了
            self.playing = None


if __name__ == '__main__':
    import qdarkstyle
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow


    class testWindow(QMainWindow):
        def __init__(self, *args, **kwargs):
            super(testWindow, self).__init__(*args, **kwargs)
            self.resize(1300, 800)
            self.detect_component = DetectComponentApp()
            self.detect_component.setObjectName("detect_component")
            # window 添加widget
            self.setCentralWidget(self.detect_component)


    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = testWindow()
    window.show()
    app.exec_()
