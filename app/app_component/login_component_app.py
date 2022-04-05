from PyQt5.QtWidgets import QMainWindow

from app.service.UserService import UserService
from app.ui_component.login_component import Ui_login


class LoginComponent(QMainWindow, Ui_login):
    def __init__(self, mainWindow, *args, **kwargs):
        self.mainWindow = mainWindow
        super(LoginComponent, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.event_register()

    def event_register(self):
        self.login_button.clicked.connect(self.login)
        self.login_cancel_button.clicked.connect(self.login_cancel)

    def login(self):
        if UserService.login_check(self.username_text.toPlainText(), self.password_text.toPlainText()):
            self.close()
            self.mainWindow.show()
        else:
            self.login_warn.setText("用户名或密码不正确")

    def login_cancel(self):
        self.close()


if __name__ == '__main__':
    import qdarkstyle
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = LoginComponent()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    # run
    window.show()
    sys.exit(app.exec_())
