from app.process_module.base.base_module import *


class CaptureModule(BaseModule):

    def __init__(self, time_threshold, capture_frame_function):
        super(CaptureModule, self).__init__()
        self.frame_counter = 0
        self.time_threshold = time_threshold
        self.frame_counter_threshold = -1  # 每多少帧抓拍一次
        self.default_fps = 30
        self.capture_frame_function = capture_frame_function

    def process_data(self, data):
        # 之后，这里检测的应该都是行李
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
