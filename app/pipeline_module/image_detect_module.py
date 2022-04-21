from abc import ABC
from pathlib import Path

import cv2
import numpy as np
import torch

from app.pipeline_module.base.base_module import *
from utils.augmentations import letterbox
from utils.general import non_max_suppression
from detect import *

opt = parse_opt()


class YoloV5DetectModule(BaseModule):
    def __init__(self, skippable=True):
        super(YoloV5DetectModule, self).__init__(skippable=skippable)
        self._load_module()

    def _load_module(self,
                     weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
                     imgsz=640,  # inference size (pixels)
                     device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                     half=False,  # use FP16 half-precision inference
                     ):
        # Load model
        w = str(weights[0] if isinstance(weights, list) else weights)
        classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

        self.model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(self.model.stride.max())  # model stride
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if half:
            self.model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
        opt['imgsz'] = check_img_size(imgsz, s=stride)  # check image size
        opt['stride'] = stride

    def _pre_process_frame(self,
                           imgs,
                           stride,
                           imgsz,  # inference size (pixels)
                           ):

        # Letterbox
        img0 = imgs.copy()
        img = [letterbox(x, imgsz, stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)
        return img, img0

    def detect(self,
               frames,
               device,
               half,  # 是否使用Float16
               conf_thres,
               iou_thres,
               classes,
               agnostic_nms,
               max_det,
               stride,
               imgsz,
               augment=False,
               visualize=False
               ):

        half &= device.type != 'cpu'  # 如果设备为gpu，使用Float16
        img, img0 = self._pre_process_frame(frames, stride, imgsz, augment)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32

        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        pred = self.model(img, augment, visualize)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        return pred

    def process_data(self, data):
        super(YoloV5DetectModule, self).process_data(data)

    def pre_run(self):
        super(YoloV5DetectModule, self).pre_run()


if __name__ == '__main__':
    detect_module = YoloV5DetectModule(opt)
