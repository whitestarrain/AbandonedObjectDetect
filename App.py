from app.app_component.abandoned_object_detect_app import AbandonedObjectDetectApp
from app.app_component.login_component_app import LoginComponent
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
import sys
import qdarkstyle

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = AbandonedObjectDetectApp()
    login_window = LoginComponent(main_window)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    # run
    login_window.show()
    sys.exit(app.exec_())
