from app.pipeline_module.base.base_module import BaseModule, StageNode
from app.pipeline_module.base.balancer import ModuleBalancer
from app.pipeline_module.base.stage_node import *


class TaskSolution(object):

    def __init__(self):
        self.modules = []
        # stage 链表头
        self.start_stage: StageNode = StageNode()
        # 记录最新的stage，链表尾
        self.last_stage: StageNode = self.start_stage
        self.source_module = None
        # 所有module 公用一个 balancer
        self.balancer = ModuleBalancer()

    def set_source_module(self, source_module):
        source_module.balancer = self.balancer
        self.source_module = source_module
        source_module.stage_node = self.start_stage
        return self

    def set_next_module(self, next_module: BaseModule):
        next_module.balancer = self.balancer
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

    def wait_for_end(self):
        self.source_module.wait_for_end()
        for module in self.modules:
            module.wait_for_end()

    def close(self):
        self.source_module.close()
        for module in self.modules:
            module.close()
