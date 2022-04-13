import numpy as np
import torch
from app.pipeline_module.core.base_module import *


class ObjectDetectModule(BaseModule):
    def __init__(self, skippable=True):
        super(ObjectDetectModule, self).__init__(skippable=skippable)

    def process_data(self, data):
        super(ObjectDetectModule, self).process_data(data)

    def pre_run(self):
        super(ObjectDetectModule, self).pre_run()


class ObjectDetectVisModule(DataDealerModule):

    def __init__(self, push_frame_func, interval=0.06, skippable=False):
        super(ObjectDetectVisModule, self).__init__(push_frame_func, interval, skippable)
        self.last_time = time.time()
        self.push_frame_func = push_frame_func
        self.last_data = None
        self.interval = interval
        self.size_waiting = True

        self.queue_threshold = 10

    def deal_skipped_data(self, data: DictData, last_data: DictData) -> DictData:
        frame = data.frame
        data = last_data
        data.skipped = None
        data.frame = frame
        # data.detections = data.detections.clone()
        # 添加抖动
        # data.detections[:, :4] += torch.rand_like(data.detections[:, :4]) * 3
        return data

    def draw_frame(self, data, fps):
        pass
