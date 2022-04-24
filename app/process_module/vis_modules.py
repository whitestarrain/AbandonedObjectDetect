import copy
import time
from abc import abstractmethod
from queue import Empty

from app.process_module.base.base_module import BaseModule, STAGE_DATA_OK, DataPackage

box_color = (0, 255, 0)
cheating_box_color = (0, 0, 255)
draw_keypoints_default = False


class DataDealerModule(BaseModule):
    def __init__(self, push_frame_func, interval=0.06, skippable=False):
        super(DataDealerModule, self).__init__(skippable=skippable)
        self.last_time = time.time()
        self.push_frame_func = push_frame_func
        self.last_data = None
        self.interval = interval
        self.size_waiting = True

        #
        self.queue_threshold = 10

    @abstractmethod
    def deal_skipped_data(self, data: DataPackage, last_data: DataPackage) -> DataPackage:
        pass

    @abstractmethod
    def draw_frame(self, data, fps):
        pass

    def process_data(self, data):
        if hasattr(data, 'skipped') and self.last_data is not None:
            data = self.deal_skipped_data(data, copy.copy(self.last_data))
        else:
            self.last_data = data
        current_time = time.time()
        interval = (current_time - self.last_time)
        fps = 1 / interval
        data.fps = fps
        self.draw_frame(data, fps=fps)
        data.interval = interval
        self.last_time = current_time  # 更新时间
        self.push_frame_func(data)
        if hasattr(data, 'source_fps'):
            time.sleep(1 / data.source_fps * (1 + self.self_balance_factor()))
        else:
            time.sleep(self.interval)
        return STAGE_DATA_OK

    def self_balance_factor(self):
        factor = max(-0.999, (self.queue.qsize() / 20 - 0.5) / -0.5)
        # print(factor)
        return factor

    def product_stage_data(self):
        # print(self.queue.qsize(), self.size_waiting)
        if self.queue.qsize() == 0:
            self.size_waiting = True
        if self.queue.qsize() > self.queue_threshold or not self.size_waiting:
            self.size_waiting = False
            try:
                stage_data = self.queue.get(block=True, timeout=1)
                return stage_data
            except Empty:
                return self.ignore_stage_data
        else:
            time.sleep(1)
            return self.ignore_stage_data

    def put_stage_data(self, stage_data):
        if self.queue is None:
            return
        self.queue.put(stage_data)

    def pre_run(self):
        super(DataDealerModule, self).pre_run()
        pass


class ObjectDetectVisModule(DataDealerModule):

    def __init__(self, push_frame_func, interval=0.06, skippable=False):
        super(ObjectDetectVisModule, self).__init__(push_frame_func, interval, skippable)

    def deal_skipped_data(self, data: DataPackage, last_data: DataPackage) -> DataPackage:
        frame = data.frame
        data = last_data
        data.skipped = None
        data.frame = frame
        return data

    def draw_frame(self, data, fps):
        frame = data.frame.copy()
        data.frame_anno = frame  # 保存绘制过的图像
