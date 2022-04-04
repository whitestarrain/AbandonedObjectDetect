from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtGui
from app.ui_component.abandoned_object_detect import Ui_AbandonedObjectDetect
from app.app_component.detect_component_app import DetectComponentApp


class AbandonedObjectDetectApp(QMainWindow, Ui_AbandonedObjectDetect):
    def __init__(self, *args, **kwargs):
        super(AbandonedObjectDetectApp, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.detect_component_app = DetectComponentApp()
        self.setCentralWidget(self.detect_component_app)

        # self.tabWidget.addTab(self.detect_component_app, "遗留物检测")
        # tabwidget设置
        # self.tabWidget.setCurrentWidget(self.detect_component_app)

        # def current_tab_change(idx, self=self):
        #     if self.current_tab_widget is not None:
        #         self.current_tab_widget.close()
        #     self.current_tab_widget = self.tabWidget.widget(idx)
        #     self.current_tab_widget.open()
        #
        # self.tabWidget.currentChanged.connect(current_tab_change)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.detect_component_app.close()
        super(AbandonedObjectDetectApp, self).closeEvent(a0)


if __name__ == '__main__':
    import qdarkstyle
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = AbandonedObjectDetectApp()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    # run
    window.show()
    sys.exit(app.exec_())
