import sys
import time

import numpy as np

from app.conf import person_class, baggage_classes
from app.process_module.base.base_module import *
from app.process_module.base.stage import StageDataStatus
from app.utils import LimitQueue, get_point_center_distance


class TimeSequenceAnalyzeList(object):
    """
    每一个物品对应需要维护的一个时间序列的检测结果
    因为有移动距离分析，所以就算出现了鸠占鹊巢情况，也没问题

    最终不会判断为遗留物的情况：
        1. 物品短暂进入
        2. 物品一直移动

    判断为遗留物的情况：
        1. 物品停留fps*analyze_period帧

    判断为遗留物之后的情况：
        1. 被短暂遮挡，需要依旧判断为遗留物
        2. 被长时间遮挡，判断为非遗留物 # 遗留物消失（被遮挡）一定时间后，才判断为遗留物消失
        3. 遗留物被移动，判断为非遗留物
    """

    def __init__(self, fps, analyse_period, factor=1 / 4, max_movement_factor=1, distance_calculate_time_interval=0.1):
        super(TimeSequenceAnalyzeList, self).__init__()
        self.factor = factor  # 被遮挡超过多长时间后，不会被判断为遗留物
        self.analyse_period = analyse_period
        self.fps = fps
        self.max_movement_factor = max_movement_factor  # 最长移动距离因数
        self.distance_calculate_time_interval = distance_calculate_time_interval  # 对象移动距离计算间隔

        # 此处多了个factor，一般使用后 analyze_length帧，即[factor * analyse_length:]
        # 如果遗留物被遮挡，检测不到的时候，会开始pop。 此时分析则是通过[m:buffer_length-m](m<factor * analyze_length)
        # 这样会有 factor * analyze_length帧的容错时间。
        self.analyze_length = int(fps * analyse_period)  # 需要多少帧进行判断
        self.buffer_length = int(fps * analyse_period * (1 + self.factor))

        self.pred_list = LimitQueue(self.buffer_length)  # 该物品的检测序列

        # 容忍度，变为0之后会删除，表示检测序列失效。
        self.lose_anno_tolerate = int(self.analyze_length * self.factor)

        self.cls = None

        self.init_time = time.time()
        self.update_time = self.init_time
        self.abandoned_judged_timestamp = None  # 被判断为遗留物的时间

    def __len__(self):
        return len(self.pred_list)

    def __getitem__(self, item):
        return self.pred_list.__getitem__(item)

    def add_pred_item(self, pred_item):
        self.pred_list.append(pred_item)
        self.update_time = time.time()
        if self.cls is None:
            self.cls = int(pred_item[5])

    def get_latest_pred(self):
        return self.pred_list[len(self.pred_list) - 1]

    def lose_anno(self):
        """
        有一帧没有检测到。

        :return:
        """
        self.lose_anno_tolerate -= 1
        # 删除最旧的检测记录。针对行李只出现一瞬间的情况
        # 如果一直出现，则不会执行到这一步
        return self.pop(0)

    def can_analyse(self):
        return len(self.pred_list) >= self.fps * self.analyse_period

    def pop(self, i):
        if len(self.pred_list) == 0:
            return None
        return self.pred_list.pop(i)

    def is_need_discard(self, timestamp):
        """
        当一个物品的时间序列队列不足1/2，同时此时又有1/4次没有检测到，针对遗留物移动出去的情况 (参数调节通过factor)
        是为了既不会因为偶尔的遮挡导致跟踪不到，同时可以及时清除无用的物品检测序列
        :param timestamp:
        :return:
        """
        if self.update_time < timestamp:  # 更新过的序列，update_time应该大于timestamp
            self.lose_anno_tolerate -= 1
            # 没有插入新检测结果
            self.lose_anno()

        if len(self.pred_list) == 0:
            return True
        elif len(self) < self.buffer_length / 2 and self.lose_anno_tolerate <= 0:
            return True

    def is_abandoned_object(self, pred_person):
        """
        针对当前物品进行分析
        移动距离小于物品长与宽，判断为遗留物
        :return:
        """

        if not self.can_analyse():
            return False

        # 默认0.1s 计算一次移动距离
        distance_compute_per_frame_num = int(self.fps * self.distance_calculate_time_interval)
        i = len(self.pred_list) - 1
        moving_distance = 0
        while i >= len(self.pred_list) - self.analyze_length + distance_compute_per_frame_num:
            moving_distance += get_point_center_distance(self.pred_list[i],
                                                         self.pred_list[i - distance_compute_per_frame_num])
            i = i - distance_compute_per_frame_num

        last_pred = self.get_latest_pred()

        max_height_or_width = max(abs(last_pred[0] - last_pred[2]), abs(last_pred[1] - last_pred[3]))

        # 移动距离过大
        if moving_distance > max_height_or_width * self.max_movement_factor:
            return False

        # 如果周围有人，则不为遗留物。除非之前判断为过遗留物
        if self.abandoned_judged_timestamp is not None:
            self.abandoned_judged_timestamp = time.time()
            return True

        # 计算两个预测框在x与y方向上的分别的最短距离
        min_person_x = sys.maxsize
        min_person_y = sys.maxsize
        for p in pred_person:
            for i in range(4):
                if i % 2 == 0:
                    min_person_x = min(min_person_x,
                                       min(abs(last_pred[i] - p[0]), abs(last_pred[i] - p[2])))
                else:
                    min_person_y = min(min_person_y,
                                       min(abs(last_pred[i] - p[1]), abs(last_pred[i] - p[3])))
        # x和y方向上的最短距离中的较大值
        min_person_instance = max(min_person_x, min_person_y)

        # 和人的距离过近
        if min_person_instance < max_height_or_width:
            return False

        self.abandoned_judged_timestamp = time.time()
        return True


class AbandonedObjectAnalysesModule(BaseModule):
    """
    遗留物分析
    关注人和行李进行分析
    时间序列上的聚类算法。哪一帧缺失也不会影响计算结果。主要逻辑就是刻画上一帧到当前帧，遗留物的移动轨迹
    opencv跟踪，可能会出现过多行李，影响计算速度。
    """

    def __init__(self, add_list_item, skippable=True, analyze_period=5):
        """

        :param skippable:
        :param analyze_period: 静止多长时间会被判断为遗留物
        """
        super(AbandonedObjectAnalysesModule, self).__init__(skippable=skippable)
        self.analyze_period = analyze_period  # 判断为遗留物的时间段
        self.analyze_length = None  # analyze_period  * fps
        self.fps = None
        self.object_pred_seq = list()
        self.frame_skip = 0  # 跳帧操作
        self.add_list_item_signal = add_list_item

    @staticmethod
    def filter_baggage_and_person(data):
        pred = data.pred
        pred_baggage = []
        pred_person = []
        pred = pred[0]  # 取第一张图片检测结果
        dim0 = len(pred)
        for i in range(dim0):
            cls = int(pred[i][5])
            label = str(data.names[cls])
            if label in baggage_classes:
                pred_baggage.append(pred[i])
            if label in person_class:
                pred_person.append(pred[i])

        return np.array([i.numpy() for i in pred_baggage if len(pred_baggage) > 0]), \
               np.array([i.numpy() for i in pred_person if len(pred_person) > 0])

    def put_into_nearest_point(self, pred_baggages):
        """
        将pred中的结果放到self.object_pred_list 中

        两种情况： 1. 判断为遗留物的物品被拿走
                 2. 行李只在镜头中停留或者移动一瞬间。

        :param pred_baggages:
        :return:
        """
        if self.analyze_length is None:
            raise Exception("没有确定判定时间或fps")

        timestamp = time.time()

        # 同类的遗留物构建进行分析
        # 同类的遗留物中，根据每帧前后的距离，判断是否是同一物品。两帧之间，物品的距离移动不得超过 一定值
        for baggage_anno in pred_baggages:
            if len(self.object_pred_seq) == 0:
                seq = TimeSequenceAnalyzeList(self.fps, self.analyze_period)
                seq.add_pred_item(baggage_anno)
                self.object_pred_seq.append(seq)

            nearest_index = -1
            nearest_distance = sys.maxsize
            for i in range(len(self.object_pred_seq)):
                time_seq = self.object_pred_seq[i]
                # 非同一类跳过
                if int(baggage_anno[5]) != time_seq.cls:
                    continue

                last_pred = time_seq.get_latest_pred()
                # 距离判断远近
                distance = get_point_center_distance(baggage_anno[:4], last_pred[:4])
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_index = i

            if -1 == nearest_distance or nearest_distance > max(abs(baggage_anno[0] - baggage_anno[2]),
                                                                abs(baggage_anno[1] - baggage_anno[3])):
                # 最短距离也太长，与上一个任何一个没有对应的，新建自己的序列
                seq = TimeSequenceAnalyzeList(self.fps, self.analyze_period)
                seq.add_pred_item(baggage_anno)
                self.object_pred_seq.append(seq)

            else:
                # 否则，最近的存入time_sequence_analyze
                self.object_pred_seq[nearest_index].add_pred_item(baggage_anno)

        if len(self.object_pred_seq) == 0:
            return

        for i in range(len(self.object_pred_seq)):
            try:
                # 多线程处理，此处可能会为有out of index 错误
                seq = self.object_pred_seq[i]
            except Exception as e:
                return
            if seq.is_need_discard(timestamp):
                self.object_pred_seq.pop(i)

    def analyse_abandoned_object(self, pred_person):
        """
        时间序列上，同类遗留物
        :return:
        """
        abandon_objects = []

        for seq in self.object_pred_seq:
            add_item_flag = False
            if seq.abandoned_judged_timestamp is None:
                add_item_flag = True
            if seq.is_abandoned_object(pred_person):
                abandon_objects.append(seq.get_latest_pred())
                if add_item_flag:
                    self.add_list_item_signal()

        return abandon_objects

    def process_data(self, data):
        # pred_baggage记录在 object_pred_list ,pred_person不进行记录
        pred_baggage, pred_person = self.filter_baggage_and_person(data)

        if self.fps is None:
            self.fps = data.source_fps
            self.analyze_length = int(self.fps * self.analyze_period)

            self.time_sequence_analyze = LimitQueue(self.analyze_length)

        self.put_into_nearest_point(pred_baggage)

        data.analyse_result = self.analyse_abandoned_object(pred_person)
        return StageDataStatus.STAGE_DATA_OK

    def pre_run(self):
        super(AbandonedObjectAnalysesModule, self).pre_run()


def _test_analyses_module():
    analyse_module = AbandonedObjectAnalysesModule()


if __name__ == '__main__':
    # _test_detect_module()
    _test_analyses_module()
