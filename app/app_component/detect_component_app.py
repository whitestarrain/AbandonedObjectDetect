import os
import time
from threading import Lock
from typing import List

import numpy as np
from PyQt5 import QtCore

from app.app_component.base_component.widget_component import *
from app.app_component.base_component.widget_component import CaptureListItem
from app.entry.video_resource import VideoResource
from app.process_module.analyze_module import AbandonedObjectAnalysesModule
from app.process_module.base.base_module import *
from app.process_module.base.data_process_pipe import *
from app.process_module.image_detect_module import YoloV5DetectModule
from app.process_module.util_module import CaptureModule,VideoSaveModule
from app.process_module.vis_modules import ObjectDetectVisModule
from app.service.video_resource_service import VideoResourceService
from app.ui_component.detect_component import Ui_DetectComponent
from app.utils import second2str

SOURCE_DIR_RELATIVE = "datasets/test_dataset"
yolov5_weight = './weights/yolov5s.torchscript.pt'
device = 'cpu'


class DetectComponentApp(QWidget, Ui_DetectComponent):
    push_frame_signal = QtCore.pyqtSignal(DataPackage)
    capture_frame_signal = QtCore.pyqtSignal(DataPackage)
    abandoned_object_list_add_signal = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(DetectComponentApp, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.video_source = 0
        self.frame_data_list = BufferList(max_length=50)
        self.process_pipe_line = None
        self.playing_thread = None
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
        self.show_box.clicked.connect(self.draw_box_flag_controller)
        self.show_person_box.clicked.connect(self.draw_box_flag_controller)

        # 自定义信号
        self.push_frame_signal.connect(self.push_frame)
        self.capture_frame_signal.connect(self.capture_frame)
        self.abandoned_object_list_add_signal.connect(self.add_abandoned_item)

    def widget_init(self):
        """
        widget 初始化
        """
        self.init_video_file_source()  # 初始化视频文件源
        self.init_video_camera_source()  # 初始化摄像源

    def init_video_file_source(self):
        videos: List[VideoResource] = VideoResourceService.get_resource_by_type(0)
        if (len(videos) == 0):
            return
        for file in videos:
            QListWidgetItemForVideoSource(self.video_resource_file_list, file.file_name, file.source_path).add_item()

    def init_video_camera_source(self):
        # 添加视频通道
        videos: List[VideoResource] = VideoResourceService.get_resource_by_type(1)
        if (len(videos) == 0):
            return
        for file in videos:
            QListWidgetItemForVideoSource(self.video_resource_list, file.file_name, file.source_path).add_item()

    def open_source(self, source):
        if not os.path.exists(source):
            source = int(source)
        self.open_source_lock.acquire(blocking=True)
        if self.process_pipe_line is not None:
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
            self.process_pipe_line = DataProcessPipe() \
                .set_source_module(VideoSourceModule(source, fps=fps)) \
                .set_next_module(YoloV5DetectModule(skippable=False)) \
                .set_next_module(AbandonedObjectAnalysesModule(lambda: self.abandoned_object_list_add_signal.emit(),
                                                               analyze_period=1)) \
                .set_next_module(CaptureModule(1, lambda d: self.capture_frame_signal.emit(d))) \
                .set_next_module(ObjectDetectVisModule(lambda d: self.push_frame_signal.emit(d))) \
                # .set_next_module(VideoSaveModule("D:/"))
            self.process_pipe_line.start()
            self.open_source_lock.release()

        Thread(target=open_source_func, args=[self]).start()

    def close_source(self):
        if self.process_pipe_line is not None:
            self.stop_playing()
            self.process_pipe_line.close()
            self.process_pipe_line = None
            self.frame_data_list.clear()
            self.video_process_bar.setMaximum(-1)

    def capture_frame(self, data):
        """
        抓拍图片
        :param frame:
        :param video_time:
        :return:
        """
        CaptureListItem(self.capture_image_list, data.frame, data.frame_counter / data.source_fps).add_item()

    def push_frame(self, data):
        try:
            # 添加帧到视频帧列表
            self.frame_data_list.append(data)
            self.video_process_bar.setMinimum(self.frame_data_list.min_index())
            self.video_process_bar.setMaximum(self.frame_data_list.max_index())

        except Exception as e:
            print(e)

    def playing_video(self):
        try:
            while self.playing_thread is not None:
                current_frame = self.video_process_bar.value()
                max_frame = self.video_process_bar.maximum()
                if current_frame < 0:
                    current_frame = 0
                elif current_frame <= max_frame:
                    data = self.frame_data_list[current_frame]
                    time.sleep(1 / data.source_fps)  # 该句一定要放在下面的if之上。否则current_frame==max_frame时，cpu空转，导致浪费cpu资源
                    if current_frame < max_frame:
                        self.video_process_bar.setValue(current_frame + 1)
                else:
                    self.stop_playing()
        except Exception as e:
            print(e)

    def play_video(self):
        if self.playing_thread is not None:
            return
        self.playing_thread = Thread(target=self.playing_video, args=())
        self.playing_thread.start()

    def change_frame(self):
        try:
            if len(self.frame_data_list) == 0:
                return
            current_frame = self.video_process_bar.value()
            max_frame = self.video_process_bar.maximum()
            # 更新界面
            data = self.frame_data_list[current_frame]
            maxData = self.frame_data_list[max_frame]
            frame = data.frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.video_screen.width() - 9, self.video_screen.height() - 9))  # 调整图像大小
            image_height, image_width, image_depth = frame.shape
            frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                           image_width * image_depth,
                           QImage.Format_RGB888)
            self.video_screen.setPixmap(QPixmap.fromImage(frame))
            # 显示时间
            current_time_process = second2str(data.frame_counter / data.source_fps)
            max_time_process = second2str(maxData.frame_counter / maxData.source_fps)

            self.time_process_label.setText(f"{current_time_process}/{max_time_process}")
        except Exception as e:
            print(e)

    def stop_playing(self):
        self.playing_thread = None

    def close(self):
        print("closing:", self)
        self.close_source()
        super(DetectComponentApp, self).close()

    def draw_box_flag_controller(self):
        for module in self.process_pipe_line.modules:
            if isinstance(module, ObjectDetectVisModule):
                setattr(module, "show_box", self.show_box.isChecked())
                setattr(module, "show_person_box", self.show_person_box.isChecked())
                return

    def add_abandoned_item(self):
        text = "%s 检测到遗留物" % time.strftime("%Y{y}%m{m}%d{d} %H:%M:%S").format(y="年", m="月", d="日")
        self.abandoned_object_ist.insertItem(0, QListWidgetItem(text, self.abandoned_object_ist))


if __name__ == '__main__':
    import qdarkstyle
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow
    from app.process_module.base.stage import DataPackage


    def _capture_test():
        class testWindow(QMainWindow):
            def __init__(self, *args, **kwargs):
                super(testWindow, self).__init__(*args, **kwargs)
                self.resize(1300, 800)
                self.detect_component = DetectComponentApp()
                data = DataPackage()
                setattr(data, "frame", cv2.imread(r"D:\MyRepo\AbandonedObjectDetect\data\images\bus.jpg"))
                setattr(data, "frame_counter", 60)
                setattr(data, "source_fps", 30)
                self.detect_component.capture_frame(data)
                self.detect_component.setObjectName("detect_component")
                # window 添加widget
                self.setCentralWidget(self.detect_component)
                item = QListWidgetItem("test", self.detect_component.abandoned_object_ist)
                self.detect_component.abandoned_object_ist.insertItem(0, item)

        app = QApplication(sys.argv)
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        window = testWindow()
        window.show()
        app.exec_()


    _capture_test()
