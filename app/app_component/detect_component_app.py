from PyQt5.QtWidgets import QWidget
from app.ui_component.detect_component import Ui_DetectComponent


class DetectComponentApp(QWidget, Ui_DetectComponent):
    def __init__(self, *args, **kwargs):
        super(DetectComponentApp, self).__init__(*args, **kwargs)
        self.setupUi(self)

    pass


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

    pass
