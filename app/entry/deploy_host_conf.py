from sqlalchemy import Column, INT, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DeployHostConf(Base):
    __tablename__ = "deploy_host_conf"

    id = Column(INT, primary_key=True, autoincrement=True)
    host = Column(String(50), nullable=False)
    port = Column(INT, nullable=False)
    uri = Column(String(30), nullable=False)
    status = Column(INT, nullable=False, default=0)
    insert_time = Column(DateTime, nullable=False)
    update_time = Column(DateTime, nullable=False)
