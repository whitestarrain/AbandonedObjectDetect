from PyQt5.QtWidgets import QMainWindow
from app.ui_component.login_component import Ui_login

class LoginComponent(QMainWindow,Ui_login):
    def __init__(self,*args,**kwargs):
        super(LoginComponent,self).__init__(*args,**kwargs)
        self.setupUi(self)



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
