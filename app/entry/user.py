from sqlalchemy import Column, INT, String, create_engine, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = "user"

    id = Column(INT, primary_key=True, autoincrement=True)
    username = Column(String(30))
    password = Column(String(30))
    insert_time = Column(DateTime, nullable=False)
    update_time = Column(DateTime, nullable=False)


if __name__ == '__main__':
    # 初始化数据库连接:
    engine = create_engine('mysql+mysqlconnector://root:root@localhost:3306/abandoned_object_detect')
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    # new_user = User(username="root_bak", password="root_bak")
    # session.add(new_user)
    # users = session.query(User).filter(username = "root_bak").all()
    print(type(session))
    # for user in users:
    #     print(user.id)
