import os


def filter_data_by_classes(classes: list, image_path, label_path):
    labels = os.listdir(label_path)
    for label in labels:
        if label == ".DS_Store":
            continue
        with open(os.path.join(label_path, label), "r+") as f:
            append_content = []
            lines = f.readlines()
            lines = [line for line in lines if classes.__contains__(int(line.split(" ")[0]))]

            if len(lines) == 0:
                f.close()
                os.remove(os.path.join(label_path, label))
                os.remove(os.path.join(image_path, label.split(".")[0] + ".jpg"))
                continue

            lines = [str(classes.index(int(line.split(" ")[0]))) + " " + str.join(" ", line.split(" ")[1:])
                     for line in lines]

            f.seek(0)
            f.truncate(0)
            f.write(str.join("", lines))


if __name__ == '__main__':
    filter_data_by_classes([0, 24, 26, 28],
                           r".\datasets\coco128\images\train2017",
                           r".\datasets\coco128\labels\train2017")
