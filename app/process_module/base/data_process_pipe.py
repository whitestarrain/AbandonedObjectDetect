from app.process_module.base.base_module import BaseModule
from app.process_module.base.stage import *


class DataProcessPipe(object):

    def __init__(self):
        self.modules = []
        # stage 链表头
        self.start_stage: StageNode = StageNode()
        # 记录最新的stage，链表尾
        self.last_stage: StageNode = self.start_stage
        self.source_module = None

    def set_source_module(self, source_module):
        self.source_module = source_module
        source_module.stage_node = self.start_stage
        return self

    def set_next_module(self, next_module: BaseModule):
        self.modules.append(next_module)
        next_stage = StageNode()
        # 链表 stage，指向下一个module
        self.last_stage.next_module = next_module
        self.last_stage.next_stage = next_stage
        self.last_stage = next_stage
        return self

    def start(self):
        for module in self.modules:
            print(f'starting modules {module}')
            module.start()
        self.source_module.start()

    def close(self):
        self.source_module.close()
        for module in self.modules:
            module.close()
