import os

from app.process_module.base.base_module import *
from app.process_module.base.stage import StageDataStatus
import cv2
import time


class CaptureModule(BaseModule):

    def __init__(self, time_threshold, capture_frame_function):
        super(CaptureModule, self).__init__()
        self.frame_counter = 0
        self.time_threshold = time_threshold
        self.frame_counter_threshold = -1  # 每多少帧抓拍一次
        self.default_fps = 30
        self.capture_frame_function = capture_frame_function

    def process_data(self, data):
        pred = data.analyse_result
        fps = data.source_fps
        if self.frame_counter_threshold < 0:
            if fps is not None:
                self.frame_counter_threshold = int(fps) * self.time_threshold
            else:
                self.frame_counter_threshold = self.default_fps * self.time_threshold

        self.frame_counter += 1

        if len(pred) == 0:
            # 没有行李的时候不进行抓拍
            return

        if self.frame_counter > self.frame_counter_threshold:
            self.frame_counter = 0
            self.capture_frame_function(data)

    def pre_run(self):
        super(CaptureModule, self).pre_run()


class VideoSaveModule(BaseModule):
    def __init__(self,out_dir, fps=30):
        super(VideoSaveModule, self).__init__()
        self.out = None
        self.fourcc = None
        self.frame = None
        self.fps = fps
        self.out_dir = out_dir

    def process_data(self, data):
        if data is None:
            return
        frame = data.frame
        height = len(frame)
        width = len(frame[0])
        if self.out is None:
            self.out = cv2.VideoWriter(str(os.path.join(self.out_dir, str(time.time()))) + ".avi", self.fourcc,
                                       self.fps, (width, height))
        self.out.write(frame)
        return StageDataStatus.STAGE_DATA_OK

    def close(self):
        self.out.release()
        self.running = False

    def pre_run(self):
        super(VideoSaveModule, self).pre_run()

        if self.out is not None:
            self.out.release()

        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
