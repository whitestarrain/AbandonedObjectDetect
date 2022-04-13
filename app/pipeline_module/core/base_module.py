import copy
import time
from abc import ABC, abstractmethod

import torch

single_process = True
from queue import Empty

if single_process:
    from queue import Queue
    from threading import Thread, Lock
else:
    from torch.multiprocessing import Queue, Lock
    from torch.multiprocessing import Process as Thread

    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

TASK_DATA_OK = 0
TASK_DATA_CLOSE = 1
TASK_DATA_IGNORE = 2
TASK_DATA_SKIP = 3
BALANCE_CEILING_VALUE = 50


class DictData(object):
    def __init__(self):
        self.frame = None
        self.fps = None
        self.detections = []
        pass


class ModuleBalancer:
    def __init__(self):
        self.max_interval = 0
        self.short_stab_interval = self.max_interval
        self.short_stab_module = None
        self.lock = Lock()
        self.ceiling_interval = 0.1

    def get_suitable_interval(self, process_interval, module):
        with self.lock:
            if module == self.short_stab_module:
                self.max_interval = (process_interval + self.max_interval) / 2
                self.short_stab_interval = module.process_interval if module.skippable else self.max_interval
                return 0
            elif process_interval > self.short_stab_interval:
                self.short_stab_module = module
                self.max_interval = process_interval
                self.short_stab_interval = module.process_interval if module.skippable else self.max_interval
                return 0
            else:
                return max(min(self.max_interval - process_interval, self.ceiling_interval), 0)


class TaskData:
    def __init__(self, task_stage, task_flag=TASK_DATA_OK):
        self.data = DictData()
        self.task_stage = task_stage
        self.task_flag = task_flag


class TaskStage:
    """
    链表节点，module之间的关系通过stage链表链接
    module1 module2 module3
    ↑         ↑        ↑
    stage1->stage2->stage3
    """

    def __init__(self):
        self.next_stage = None
        self.next_module = None

    def to_next_stage(self, task_data: TaskData):
        self.next_module.put_task_data(task_data)
        task_data.task_stage = self.next_stage


class BaseModule(ABC):
    def __init__(self, balancer=None, skippable=True, queueSize=50):
        self.worker_thread = None
        self.running = False
        self.skippable = skippable
        self.ignore_task_data = TaskData(task_stage=None, task_flag=TASK_DATA_IGNORE)
        self.queue = Queue(maxsize=queueSize)
        self.balancer: ModuleBalancer = balancer
        self.process_interval = 0.01
        self.process_interval_scale = 1
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
        print(f'closing: {self}')
        self.running = False

    def _run(self):
        """
        核心执行程序。
        pre_run():进行准备，开启资源等
        product_task_data():从队列中取出资源
        task_stage.to_next_stage(task_data):把数据传到下一个模块
        pre_close():结束执行前清理资源
        """
        self.running = True
        self.pre_run()
        while self.running:
            task_data = self.product_task_data()

            execute_condition = (task_data.task_flag == TASK_DATA_OK) or \
                                (task_data.task_flag == TASK_DATA_SKIP and not self.skippable)

            start_time = time.time()
            execute_result = self.process_data(task_data.data) if execute_condition else task_data.task_flag
            process_interval = min((time.time() - start_time) * self.process_interval_scale, BALANCE_CEILING_VALUE)
            task_stage = task_data.task_stage
            if execute_result == TASK_DATA_SKIP:
                task_data.task_flag = TASK_DATA_SKIP
            else:
                self.process_interval = process_interval
            if execute_result == TASK_DATA_IGNORE:
                continue
            else:
                if execute_result == TASK_DATA_CLOSE:
                    task_data.task_flag = TASK_DATA_CLOSE
                    self.close()
                if task_stage.next_stage is not None:
                    task_stage.to_next_stage(task_data)
            if self.balancer is not None:
                suitable_interval = self.balancer.get_suitable_interval(process_interval, self)
                if suitable_interval > 0:
                    time.sleep(suitable_interval)

    def start(self):
        p = Thread(target=self._run, args=())
        p.start()
        self.worker_thread = p
        return p

    def put_task_data(self, task_data):
        self.queue.put(task_data)
        self._refresh_process_interval_scale()

    def _refresh_process_interval_scale(self):
        self.process_interval_scale = max(self.queue.qsize(), 1)

    def product_task_data(self):
        try:
            task_data = self.queue.get(block=True, timeout=1)
            self._refresh_process_interval_scale()
            return task_data
        except Empty:
            return self.ignore_task_data

    def wait_for_end(self):
        self.worker_thread.join()


class DataDealerModule(BaseModule):
    def __init__(self, push_frame_func, interval=0.06, skippable=False):
        super(DataDealerModule, self).__init__(skippable=skippable)
        self.last_time = time.time()
        self.push_frame_func = push_frame_func
        self.last_data = None
        self.interval = interval
        self.size_waiting = True

        self.queue_threshold = 10

    @abstractmethod
    def deal_skipped_data(self, data: DictData, last_data: DictData) -> DictData:
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
        return TASK_DATA_OK

    def self_balance_factor(self):
        factor = max(-0.999, (self.queue.qsize() / 20 - 0.5) / -0.5)
        # print(factor)
        return factor

    def product_task_data(self):
        # print(self.queue.qsize(), self.size_waiting)
        if self.queue.qsize() == 0:
            self.size_waiting = True
        if self.queue.qsize() > self.queue_threshold or not self.size_waiting:
            self.size_waiting = False
            try:
                task_data = self.queue.get(block=True, timeout=1)
                return task_data
            except Empty:
                return self.ignore_task_data
        else:
            time.sleep(1)
            return self.ignore_task_data

    def put_task_data(self, task_data):
        self.queue.put(task_data)

    def pre_run(self):
        super(DataDealerModule, self).pre_run()
        pass
