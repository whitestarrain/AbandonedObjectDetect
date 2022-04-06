import cv2 as cv
from pathlib import Path
import os


def img_to_video(img_dir, output_dir):
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
    fps = 60
    out = cv.VideoWriter(str(output_dir / os.path.basename(img_dir)) + ".avi", fourcc, fps, (height, width))
    [out.write(cv.imread(str(img_dir / img))) for img in imgs]
    out.release()


def video_to_img():
    pass


if __name__ == '__main__':
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]  # YOLOv5 root directory
    source_dir = ROOT / "../datasets/S1-T1-C/video/pets2006/S1-T1-C/1"
    output_dir = ROOT / "../datasets/test_dataset"
    img_to_video(source_dir, output_dir)
