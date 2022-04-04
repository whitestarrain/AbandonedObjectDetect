from app.app_component.abandoned_object_detect_app import AbandonedObjectDetectApp
from PyQt5.QtWidgets import QApplication,QMainWindow,QWidget
import sys
import qdarkstyle

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AbandonedObjectDetectApp()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    # run
    window.show()
    sys.exit(app.exec_())
    pass
