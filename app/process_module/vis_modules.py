import copy
import time
from abc import abstractmethod
from queue import Empty

import cv2

from app.process_module.base.base_module import BaseModule
from app.process_module.base.stage import StageDataStatus, DataPackage


def draw_box_and_labels(frame, x1, y1, x2, y2, label, conf):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.8
    font_height = int(font_size * 40)
    cv2.rectangle(frame, (x1 - 2, y1), (x1 + len(label) * 15, y1 - font_height), (0, 255, 0),
                  -1)  # thickness=-1 为实心
    cv2.putText(frame, label, (x1, y1 - 10), font, font_size, (0, 0, 255), 1,
                cv2.LINE_AA)
    cv2.putText(frame, conf, (x1, y1 + font_height - 10), font, font_size, (0, 0, 255), 1,
                cv2.LINE_AA)


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
        return StageDataStatus.STAGE_DATA_OK

    def product_stage_data(self):
        try:
            stage_data = self.queue.get(block=True, timeout=1)
            return stage_data
        except Empty:
            return self.ignore_stage_data

    def put_stage_data(self, stage_data):
        if self.queue is None:
            return
        self.queue.put(stage_data)

    def pre_run(self):
        super(DataDealerModule, self).pre_run()


class ObjectDetectVisModule(DataDealerModule):

    def __init__(self, push_frame_func, interval=0.06, skippable=False):
        super(ObjectDetectVisModule, self).__init__(push_frame_func, interval, skippable)
        self.show_box = True
        self.show_person_box = True

    def deal_skipped_data(self, data: DataPackage, last_data: DataPackage) -> DataPackage:
        frame = data.frame
        data = last_data
        data.skipped = None
        data.frame = frame
        return data

    def draw_frame(self, data, fps):
        if not self.show_box:
            return

        pred = data.pred[0]
        dim0 = len(pred)
        for i in range(dim0):
            x1 = int(pred[i][0])
            y1 = int(pred[i][1])
            x2 = int(pred[i][2])
            y2 = int(pred[i][3])
            conf = "%.2f" % float(pred[i][4], )
            cls = int(pred[i][5])
            label = str(data.names[cls])
            if label == "person" and not self.show_person_box:
                continue
            draw_box_and_labels(data.frame, x1, y1, x2, y2, label, conf)
        # frame = data.frame.copy()
        # data.frame_anno = frame  # 保存绘制过的图像
