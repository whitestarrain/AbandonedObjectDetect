#!/usr/bin/env python
import cv2
import time
from app.process_module.image_detect_module import YoloV5DetectModule

if __name__ == '__main__':
    detect_module = YoloV5DetectModule(skippable=False)
    names = detect_module.names
    img = cv2.imread("data/images/bus.jpg")
    t = 0
    iter = 100
    for i in range(iter):
        t1 = time.time()
        detect_module.detect(img)
        t2 = time.time()
        t = t+ (t2-t1)

    print(f"time spend per img: {t/100}")


