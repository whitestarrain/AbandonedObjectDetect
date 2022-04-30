from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session


class DaoUtils(object):
    # 初始化数据库连接:
    engine = create_engine('mysql+mysqlconnector://root:root@localhost:3306/abandoned_object_detect')
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)

    @classmethod
    def get_session(cls) -> Session:
        return cls.DBSession()
