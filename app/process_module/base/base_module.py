import copy
import time
from abc import ABC, abstractmethod
from queue import Empty
from queue import Queue
from threading import Thread

from app.process_module.base.stage import *
from app.process_module.base.stage import StageDataStatus


class BaseModule(ABC):
    def __init__(self, skippable=True, queueSize=50):
        self.worker_thread = None
        self.running = False
        self.skippable = skippable
        self.ignore_stage_data = StageData(stage_node=None, data_status=StageDataStatus.STAGE_DATA_ABSTRACT)
        self.queue = Queue(maxsize=queueSize)
        print(f'created: {self}')

    @abstractmethod
    def process_data(self, data):
        print("BaseModule process data", self)
        pass

    @abstractmethod
    def pre_run(self):
        """
        执行前准备工作
        """
        self.running = True

    def close(self):
        """
        结束运行
        """
        # print(f'closing: {self}')
        self.running = False
        self.queue.queue.clear()
        self.queue = None

    def _run(self):
        """
        核心执行程序。
        pre_run():进行准备，开启资源等
        product_stage_data():从队列中取出资源
        stage_node.to_next_stage(stage_data):把数据传到下一个模块
        pre_close():结束执行前清理资源
        """
        self.running = True
        self.pre_run()
        while True:
            if not self.running:
                print("close module:", self)
                break
            # print("_run:", self)

            data = self.product_stage_data()

            execute_condition = (data.stage_flag == StageDataStatus.STAGE_DATA_OK) or \
                                (data.stage_flag == StageDataStatus.STAGE_DATA_SKIP and not self.skippable)

            execute_result = self.process_data(data.data) if execute_condition else data.stage_flag
            stage_node = data.stage_node
            if execute_result == StageDataStatus.STAGE_DATA_SKIP:
                data.stage_flag = StageDataStatus.STAGE_DATA_SKIP

            if execute_result == StageDataStatus.STAGE_DATA_ABSTRACT:
                continue
            elif execute_result == StageDataStatus.STAGE_DATA_CLOSE:
                data.stage_flag = StageDataStatus.STAGE_DATA_CLOSE
                self.close()
            elif stage_node.next_stage is not None:
                stage_node.to_next_stage(data)

    def start(self):
        print("run:", self)
        p = Thread(target=self._run, args=())
        p.start()
        self.worker_thread = p
        return p

    def put_stage_data(self, stage_data):
        if self.queue is None:
            return

        self.queue.put(stage_data)

    def product_stage_data(self):
        try:
            stage_data = self.queue.get(block=True, timeout=1)
            return stage_data
        except Empty:
            return self.ignore_stage_data


class BaseProcessModule(BaseModule):
    def __init__(self, push_frame_func, interval=0.06, skippable=False):
        super(BaseProcessModule, self).__init__(skippable=skippable)
        self.last_time = time.time()
        self.push_frame_func = push_frame_func
        self.last_data = None
        self.interval = interval
        self.size_waiting = True

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
        # if hasattr(data, 'source_fps'):
        #     time.sleep(1 / data.source_fps * (1 + self.self_balance_factor()))
        # else:
        #     time.sleep(self.interval)
        return StageDataStatus.STAGE_DATA_OK

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
        self.queue.put(stage_data)

    def pre_run(self):
        super(BaseProcessModule, self).pre_run()
