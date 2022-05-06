import os
from pathlib import Path

import cv2
import cv2 as cv


def img_to_video(img_dir, output_dir, fps):
    if not os.path.exists(img_dir):
        return
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    imgs = [*filter(lambda img: img.endswith("jpeg") or img.endswith("png") or img.endswith("jpg"),
                    os.listdir(img_dir))]
    imgs.sort()
    shape = cv.imread(str(img_dir / imgs[0])).shape
    width = shape[0]
    height = shape[1]
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    out = cv.VideoWriter(str(output_dir / os.path.basename(img_dir)) + ".avi", fourcc, fps, (height, width))
    [out.write(cv.imread(str(img_dir / img))) for img in imgs]
    out.release()


def video_to_img(video_path, output_dir: Path):
    if not os.path.exists(video_path):
        raise "can not find video"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise "can not load video"

    ret, frame = cap.read()
    i = 0
    while ret:
        cv.imwrite(str(output_dir / ("1%011d.jpg" % i)), frame)
        i += 1
        ret, frame = cap.read()
    cap.release()


if __name__ == '__main__':
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[2]  # YOLOv5 root directory

    # source_dir = ROOT / "datasets/S1-T1-C/video/pets2006/S1-T1-C/1"
    # output_dir = ROOT / "datasets/test_dataset"
    # img_to_video(source_dir, output_dir, 20)

    video_path = ROOT / "datasets/test_dataset/train.mp4"
    output_dir = ROOT / "datasets/my_dataset/images/train"
    video_to_img(video_path, output_dir)
