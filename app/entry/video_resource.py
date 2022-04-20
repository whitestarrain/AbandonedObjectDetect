from sqlalchemy import Column, INT, String, create_engine, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class VideoResource(Base):
    __tablename__ = "video_resource"

    id = Column(INT, primary_key=True, autoincrement=True)
    file_name = Column(String(20), nullable=False)
    source_path = Column(String(100), nullable=False)
    type = Column(INT, nullable=False)
    insert_time = Column(DateTime, nullable=False)
    update_time = Column(DateTime, nullable=False)