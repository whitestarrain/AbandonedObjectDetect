from app.service.DaoUtils import DaoUtils
from app.entry.video_resource import VideoResource
from enum import Enum


class VideoResourceType(Enum):
    video_file = 0
    camera_source = 1


class VideoResourceDao(object):
    @staticmethod
    def select_user_by_type(t):
        session = DaoUtils.get_session()
        resources = session.query(VideoResource).filter_by(type=t).all()
        session.close()
        return resources


class VideoResourceService(object):
    @staticmethod
    def get_resource_by_type(t):
        return VideoResourceDao.select_user_by_type(t)
