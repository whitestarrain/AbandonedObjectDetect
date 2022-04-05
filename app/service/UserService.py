from app.service.DaoUtils import DaoUtils
from app.entry.User import User


class UserDao(object):
    @staticmethod
    def select_user_by_Id(self, id):
        session = DaoUtils.get_session()
        users = session.query(User).filter_by(id=id).all()
        session.close()
        return users

    @staticmethod
    def select_user_by_username(username):
        session = DaoUtils.get_session()
        users = session.query(User).filter_by(username=username).all()
        session.close()
        return users

    @staticmethod
    def select_user_by_username_password(username, password):
        session = DaoUtils.get_session()
        users = session.query(User).filter_by(username=username, password=password).all()
        session.close()
        return users


class UserService(object):
    @staticmethod
    def login_check(username, password):
        if (0 == len(username) or 0 == len(password)):
            return False
        users = UserDao.select_user_by_username_password(username, password)
        if 0 == len(users):
            return False
        else:
            return True
