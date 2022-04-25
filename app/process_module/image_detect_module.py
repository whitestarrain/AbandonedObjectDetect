import cv2
import numpy as np
import torch

from app.process_module.base.base_module import *
from utils.augmentations import letterbox
from detect import parse_opt as get_opt, ROOT, select_device, Path, check_suffix, attempt_load, load_classifier, \
    check_img_size, non_max_suppression, scale_coords

from app.process_module.base.stage import StageDataStatus

opt = get_opt()


class YoloV5DetectModule(BaseModule):
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


if __name__ == '__main__':
    import numpy as np

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
