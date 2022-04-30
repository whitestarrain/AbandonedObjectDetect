from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QListWidget

from app.process_module.source_modules import *
from app.ui_component.capture_item_component import Ui_CaptureItem
from app.utils import *


class CaptureItemWidget(QWidget, Ui_CaptureItem):
    def __init__(self, parent=None):
        super(CaptureItemWidget, self).__init__(parent)
        self.setupUi(self)


class QListWidgetItemForVideoSource(QListWidgetItem):
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
        super(QListWidgetItemForVideoSource, self).__init__()
        self.list_widget: QListWidget = list_widget
        self.type = type
        self.setText(name + "(" + src + ")")
        self.src = src

    def add_item(self):
        len = self.list_widget.count()
        for i in range(len):
            if self.list_widget.item(i).src == self.src:
                return
        self.list_widget.addItem(self)


class CaptureListItem(QListWidgetItem):
    """
    抓拍列表中的item
    """

    def __init__(self, list_widget: QListWidget, frame, time):
        super(CaptureListItem, self).__init__()
        self.list_widget = list_widget
        self.item_widget = CaptureItemWidget()
        self.setSizeHint(QSize(200, 200))

        self.frame = frame
        self.time = time  # frame_count / fps

    def add_item(self):
        size = self.sizeHint()
        self.list_widget.insertItem(0, self)
        self.item_widget.setSizeIncrement(size.width(), size.height())
        self.list_widget.setItemWidget(self, self.item_widget)

        # 设置图像
        capture_img = self.item_widget.capture_img
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (capture_img.width(), capture_img.height()))
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame, image_width, image_height, image_width * image_depth, QImage.Format_RGB888)
        self.item_widget.capture_img.setPixmap(QPixmap.fromImage(frame))
        # 设置时间
        self.item_widget.capture_time.setText(f'{second2str(self.time)}')


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    item = CaptureListItem(QListWidget(), None, None)
    print(item, 1)
