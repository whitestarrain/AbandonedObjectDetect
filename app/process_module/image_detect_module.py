import json
import sys
import time
from typing import List

import cv2
import numpy as np
import requests
import torch
from sanic.response import json

from app.conf import baggage_classes, person_class
from app.process_module.base.base_module import *
from app.process_module.base.stage import StageDataStatus
from app.service.deploy_host_conf_service import DeployHostConfService
from app.utils import LimitQueue, get_point_center_distance
from detect import parse_opt as get_opt, ROOT, select_device, Path, check_suffix, attempt_load, load_classifier, \
    check_img_size, non_max_suppression, scale_coords
from utils.augmentations import letterbox

opt = get_opt()


class YoloV5DetectModule(BaseModule):
    """
    本机检测
    """

    def __init__(self, skippable=True):
        super(YoloV5DetectModule, self).__init__(skippable=skippable)
        self.parse_opt(**vars(opt))
        self._load_module()

    def parse_opt(self,
                  weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
                  source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
                  imgsz=640,  # inference size (pixels)
                  conf_thres=0.25,  # confidence threshold
                  iou_thres=0.45,  # NMS IOU threshold
                  max_det=1000,  # maximum detections per image
                  device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                  view_img=False,  # show results
                  save_txt=False,  # save results to *.txt
                  save_conf=False,  # save confidences in --save-txt labels
                  save_crop=False,  # save cropped prediction boxes
                  nosave=False,  # do not save images/videos
                  classes=None,  # filter by class: --class 0, or --class 0 2 3
                  agnostic_nms=False,  # class-agnostic NMS
                  augment=False,  # augmented inference
                  visualize=False,  # visualize features
                  update=False,  # update all models
                  project=ROOT / 'runs/detect',  # save results to project/name
                  name='exp',  # save results to project/name
                  exist_ok=False,  # existing project/name ok, do not increment
                  line_thickness=3,  # bounding box thickness (pixels)
                  hide_labels=False,  # hide labels
                  hide_conf=False,  # hide confidences
                  half=False,  # use FP16 half-precision inference
                  dnn=False,  # use OpenCV DNN for ONNX inference
                  ):
        self.weights = weights
        self.source = source
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.update = update
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.dnn = dnn

    def _load_module(self):

        self.device = select_device(self.device)
        self.half &= self.device.type != 'cpu'  # 如果设备为gpu，使用Float16

        # Load model
        w = str(self.weights[0] if isinstance(self.weights, list) else self.weights)
        classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        self.stride, self.names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

        self.model = torch.jit.load(w) if 'torchscript' in w else attempt_load(self.weights, map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=self.device)['model']).to(self.device).eval()
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

    def _pre_process_frame(self, frame):

        # Letterbox
        img0 = frame.copy()
        # img0 = cv2.flip(img0, 1)  # flip left-right

        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return img, img0

    @torch.no_grad()
    def detect(self, imgs):
        self.half &= self.device.type != 'cpu'  # 如果设备为gpu，使用Float16
        img, img0 = self._pre_process_frame(imgs)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32

        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        pred = self.model(img, self.augment, self.visualize)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
        # print(pred)

        for i, det in enumerate(pred):  # per image
            if len(det):
                # 预测到的结果是指定size图片上的。这里转换为画在原图片上的框
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    lable = self.names[c]

        return pred

    def process_data(self, data):
        data.pred = self.detect(data.frame)
        data.names = self.names
        return StageDataStatus.STAGE_DATA_OK

    def pre_run(self):
        super(YoloV5DetectModule, self).pre_run()


class YoloV5DetectRemoteModule(BaseModule):
    """
    调用detect服务器，获取检测结果
    """

    def __init__(self, skippable=True):
        super(YoloV5DetectRemoteModule, self).__init__(skippable=skippable)
        self.deploy_host_conf_service = DeployHostConfService(interval=10)
        self.deploy_host_conf_service.start_update()

    def process_data(self, data):
        host_conf = self.deploy_host_conf_service.get_host_conf()
        request_data = json.dumps(data.frame.tolist())
        response = requests.post(f"http://{host_conf.host}:{host_conf.port}{host_conf.uri}", data={
            "data": request_data
        })
        resp_data = json.loads(response._content)
        data.pred = resp_data["pred"]
        data.names = resp_data["names"]
        return StageDataStatus.STAGE_DATA_OK

    def pre_run(self):
        super(YoloV5DetectRemoteModule, self).pre_run()

    def close(self):
        super(YoloV5DetectRemoteModule, self).close()
        self.deploy_host_conf_service.stop_update()


class CaptureModule(BaseModule):

    def __init__(self, time_threshold, capture_frame_function):
        super(CaptureModule, self).__init__()
        self.frame_counter = 0
        self.time_threshold = time_threshold
        self.frame_counter_threshold = -1  # 每多少帧抓拍一次
        self.default_fps = 30
        self.capture_frame_function = capture_frame_function

    def process_data(self, data):
        # 之后，这里检测的应该都是行李
        pred = data.pred
        fps = data.source_fps
        if self.frame_counter_threshold < 0:
            if fps is not None:
                self.frame_counter_threshold = int(fps) * self.time_threshold
            else:
                self.frame_counter_threshold = self.default_fps * self.time_threshold

        self.frame_counter += 1

        if len(pred) == 0:
            # 没有行李的时候不进行抓拍
            return

        if self.frame_counter > self.frame_counter_threshold:
            self.frame_counter = 0
            self.capture_frame_function(data)

    def pre_run(self):
        super(CaptureModule, self).pre_run()


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

    def __init__(self, fps, analyse_period):
        super(TimeSequenceAnalyzeList, self).__init__()
        self.factor = 1 / 4
        self.analyse_period = analyse_period
        self.fps = fps

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
        当一个物品的时间序列队列不足1/2，同时此时又有1/4次没有检测到，针对遗留物移动出去的情况 todo: 参数调节
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

        # 0.5s 计算一次移动距离
        distance_compute_per_frame_num = int(self.fps / 2)
        i = len(self.pred_list)
        moving_distance = 0
        while i >= len(self.pred_list) - self.analyze_length + distance_compute_per_frame_num:
            moving_distance += get_point_center_distance(self.pred_list[i],
                                                         self.pred_list[i - distance_compute_per_frame_num])
            i = i - distance_compute_per_frame_num

        last_pred = self.get_latest_pred()

        max_height_or_width = max(abs(last_pred[0] - last_pred[2]), abs(last_pred[1] - last_pred[3]))
        if moving_distance > max_height_or_width:
            return False

        # 如果周围有人，则不为遗留物。除非之前判断为过遗留物
        if self.abandoned_judged_timestamp is not None:
            return True

        # 计算两个预测框在x与y方向上的最短距离
        min_person_distance = sys.maxsize
        for p in pred_person:
            for i in range(4):
                if i % 2 == 0:
                    min_person_distance = min(min_person_distance,
                                              min(abs(last_pred[i] - p[0]), abs(last_pred[i] - p[2])))
                else:
                    min_person_distance = min(min_person_distance,
                                              min(abs(last_pred[i] - p[1]), abs(last_pred[i] - p[3])))

        # 和人的距离过近
        if min_person_distance < max_height_or_width:
            return False

        return True


class AbandonedObjectAnalysesModule(BaseModule):
    """
    遗留物分析
    关注人和行李进行分析
    时间序列上的聚类算法。哪一帧缺失也不会影响计算结果。主要逻辑就是刻画上一帧到当前帧，遗留物的移动轨迹
    opencv跟踪，可能会出现过多行李，影响计算速度。
    """

    def __init__(self, skippable=True, analyze_period=5):
        super(AbandonedObjectAnalysesModule, self).__init__(skippable=skippable)
        self.analyze_period = analyze_period  # 判断为遗留物的时间段
        self.analyze_length = None  # analyze_period  * fps
        self.fps = None
        self.object_pred_seq = List[TimeSequenceAnalyzeList]()
        self.frame_skip = 0  # 跳帧操作

    @staticmethod
    def filter_baggage_and_person(pred):
        pred_baggage = []
        pred_person = []
        pred = pred[0]  # 取第一张图片检测结果
        dim0 = len(pred)
        for i in range(dim0):
            cls = int(pred[i][5])
            label = str(pred.names[cls])
            if label in baggage_classes:
                pred_baggage.append(pred[i])
            if label in person_class:
                pred_person.append(pred[i])

        return np.array(pred_baggage), np.array(pred_person)

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
                seq = TimeSequenceAnalyzeList(self.analyze_length, self.fps, self.analyze_period)
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

        for i in range(len(self.object_pred_seq)):
            seq = self.object_pred_seq[i]
            if seq.is_need_discard(timestamp):
                self.object_pred_seq.pop(i)

    def analyse_abandoned_object(self, pred_person):
        """
        时间序列上，同类遗留物
        :return:
        """
        abandon_objects = []

        for seq in self.object_pred_seq:
            if seq.is_abandoned_object(pred_person):
                abandon_objects.append(seq.get_latest_pred())

        return abandon_objects

    def process_data(self, data):
        # pred_baggage记录在 object_pred_list ,pred_person不进行记录
        pred_baggage, pred_person = self.filter_baggage_and_person(data)

        if self.fps is None:
            self.fps = data.fps
            self.analyze_length = int(self.fps * self.analyze_period)

            self.time_sequence_analyze = LimitQueue(self.analyze_length)

        self.put_into_nearest_point(pred_baggage)

        data.analyse_result = self.analyse_abandoned_object(pred_person)
        return StageDataStatus.STAGE_DATA_OK

    def pre_run(self):
        super(AbandonedObjectAnalysesModule, self).pre_run()


def _test_detect_module():
    detect_module = YoloV5DetectModule(opt)
    frame = cv2.imread(r"D:\MyRepo\AbandonedObjectDetect\data\images\bus.jpg")
    pred: torch.Tensor = detect_module.detect(frame)[0]
    print(pred)
    dim0 = len(pred)
    for i in range(dim0):
        x1 = int(pred[i][0])
        y1 = int(pred[i][1])
        x2 = int(pred[i][2])
        y2 = int(pred[i][3])
        conf = "%.2f" % float(pred[i][4], )
        cls = int(pred[i][5])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.8
        label = str(detect_module.names[cls])
        font_height = int(font_size * 40)
        cv2.rectangle(frame, (x1 - 2, y1), (x1 + len(label) * 15, y1 - font_height), (0, 255, 0),
                      -1)  # thickness=-1 为实心
        cv2.putText(frame, label, (x1, y1 - 10), font, font_size, (0, 0, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, conf, (x1, y1 + font_height - 10), font, font_size, (0, 0, 255), 1,
                    cv2.LINE_AA)

    # for one_pred in pred:
    #     for pred_index in range(len(one_pred)):
    #         one_pred[pred_index] = int(one_pred[pred_index])
    #     for x1, x2, y1, y2, conf, cls in one_pred:
    #         cv2.rectangle(frame, (int(x1), y1), (x2, y2), (0, 255, 0), 3)

    cv2.imshow("bus", frame)
    k = cv2.waitKey(0)  # show time,0: don't close
    if k == ord("q"):
        cv2.destroyAllWindows()

    #
    #
    # # create a VideoCapture object
    # cap = cv.VideoCapture(0)
    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()
    # while True:
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #     # if frame is read correctly ret is True
    #     if not ret:
    #         print("Can't receive frame (stream end?). Exiting ...")
    #         break
    #     pred = detect_module.detect(np.stack([frame]))
    #     print(pred)
    #
    #     cv.imshow('frame', frame)
    #     if cv.waitKey(1) == ord('q'):
    #         break
    # # When everything done, release the capture
    # cap.release()
    # cv.destroyAllWindows()


def _test_analyses_module():
    analyse_module = AbandonedObjectAnalysesModule()
    pass


if __name__ == '__main__':
    # _test_detect_module()
    _test_analyses_module()
