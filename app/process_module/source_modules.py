import cv2

from app.process_module.base.base_module import BaseModule
from app.process_module.base.stage import *
from app.process_module.base.stage import StageDataStatus


class VideoSourceModule(BaseModule):

    def __init__(self, source=0, fps=25, skippable=False):
        super(VideoSourceModule, self).__init__(skippable=skippable)
        self.frame_counter = 0
        self.stage_node = None  # DataProcessPipe中初始化
        self.source = source
        self.cap = None
        self.frame = None
        self.ret = False
        self.skip_timer = 0  # 跳帧
        self.set_fps(fps)
        self.loop = True

    def process_data(self, data):
        if not self.ret:
            if self.loop:
                self.pre_run()
                return StageDataStatus.STAGE_DATA_ABSTRACT
            else:
                return StageDataStatus.STAGE_DATA_CLOSE
        data.source_fps = self.fps
        data.frame = self.frame
        data.frame_counter = self.frame_counter
        self.ret, self.frame = self.cap.read()
        self.frame_counter += 1
        result = StageDataStatus.STAGE_DATA_OK
        return result

    def product_stage_data(self):
        return StageData(self.stage_node)

    def set_fps(self, fps):
        self.fps = fps
        self.interval = 1 / fps

    def close(self):
        self.cap.release()
        super().close()

    def pre_run(self):
        super(VideoSourceModule, self).pre_run()
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            self.set_fps(self.cap.get(cv2.CAP_PROP_FPS))
            self.ret, self.frame = self.cap.read()
            self.frame_counter += 1
            print("fps: ", self.fps)
