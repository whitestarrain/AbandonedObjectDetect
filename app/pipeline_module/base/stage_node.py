STAGE_DATA_OK = 0
STAGE_DATA_CLOSE = 1
STAGE_DATA_IGNORE = 2
STAGE_DATA_SKIP = 3


class DataPackage:
    """
    存放处理过程中产生的数据
    比如frame，处理过的frame，检测结果等
    """

    def __init__(self):
        self.frame = None
        self.fps = None
        self.detections = []


class StageData:
    def __init__(self, stage_node, stage_flag=STAGE_DATA_OK):
        self.data = DataPackage()
        self.stage_node = stage_node
        self.stage_flag = stage_flag


class StageNode:
    """
    链表节点，module之间的关系通过stage链表链接
    module1 module2 module3
    ↑         ↑        ↑
    stage1->stage2->stage3
    """

    def __init__(self):
        self.next_stage = None
        self.next_module = None

    def to_next_stage(self, stage_data: StageData):
        self.next_module.put_stage_data(stage_data)
        stage_data.stage_node = self.next_stage
