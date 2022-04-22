import cv2
import numpy as np
import torch

from app.pipeline_module.base.base_module import *
from utils.augmentations import letterbox
from detect import parse_opt as get_opt, ROOT, select_device, Path, check_suffix, attempt_load, load_classifier, \
    check_img_size, non_max_suppression

opt = get_opt()


class YoloV5DetectModule(BaseModule):
    def __init__(self, opt, skippable=True):
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
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

        self.model = torch.jit.load(w) if 'torchscript' in w else attempt_load(self.weights, map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=self.device)['model']).to(self.device).eval()
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size

    def _pre_process_frame(self, imgs):

        # Letterbox
        img0 = imgs.copy()
        img = [letterbox(x, self.imgsz, self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)
        return img, img0

    def pred_convert(self, pred):
        return pred

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
        return self.pred_convert(pred)

    def process_data(self, data):
        super(YoloV5DetectModule, self).process_data(data)

    def pre_run(self):
        super(YoloV5DetectModule, self).pre_run()


if __name__ == '__main__':
    detect_module = YoloV5DetectModule(opt)
    pred = detect_module.detect(np.stack(
        [
            cv2.imread(r"D:\MyRepo\AbandonedObjectDetect\data\images\bus.jpg")
        ]
    ))
    print(pred)
