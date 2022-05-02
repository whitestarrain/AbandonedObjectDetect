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
