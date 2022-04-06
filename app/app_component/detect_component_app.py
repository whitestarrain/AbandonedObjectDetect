from PyQt5.QtWidgets import QWidget, QListWidgetItem, QListWidget
import os
from pathlib import Path
from PIL import Image
import cv2 as cv
import sys
import csv
from itertools import islice
from PyQt5.QtGui import QImage, QPixmap, QIcon
from app.ui_component.detect_component import Ui_DetectComponent
from threading import Thread, Lock

SOURCE_DIR_RELATIVE = "datasets/test_dataset"


class QListWidgetItemForVideo(QListWidgetItem):
    def __init__(self, list_widget, name, src):
        super(QListWidgetItemForVideo, self).__init__()
        self.list_widget: QListWidget = list_widget
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


class DetectComponentApp(QWidget, Ui_DetectComponent):
    def __init__(self, *args, **kwargs):
        super(DetectComponentApp, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.widget_init()

    def event_register(self):
        pass

    def widget_init(self):
        """
        widget 初始化
        """
        self.init_video_camera_source()  # 初始化摄像源
        self.init_video_file_source()  # 初始化视频文件源

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
        QListWidgetItemForVideo(self.video_resource_list, "摄像头", "0").add_item()


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
