import time

import cv2

from app.pipeline_module.base.stage_node import *
from app.pipeline_module.base.base_module \
    import BaseModule, STAGE_DATA_CLOSE, STAGE_DATA_OK, StageData, STAGE_DATA_SKIP, STAGE_DATA_ABSTRACT


class VideoModule(BaseModule):

    def __init__(self, source=0, fps=25, skippable=False):
        super(VideoModule, self).__init__(skippable=skippable)
        self.stage_node = None
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
                return STAGE_DATA_ABSTRACT
            else:
                return STAGE_DATA_CLOSE
        data.source_fps = self.fps
        data.frame = self.frame
        self.ret, self.frame = self.cap.read()
        result = STAGE_DATA_OK
        if self.skip_timer != 0:
            result = STAGE_DATA_SKIP
            data.skipped = None
        skip_gap = int(self.fps * self.balancer.short_stab_interval)
        if self.skip_timer > skip_gap:
            self.skip_timer = 0
        else:
            self.skip_timer += 1
        time.sleep(self.interval)
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
        super(VideoModule, self).pre_run()
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            self.set_fps(self.cap.get(cv2.CAP_PROP_FPS))
            self.ret, self.frame = self.cap.read()
            print("视频源帧率: ", self.fps)
