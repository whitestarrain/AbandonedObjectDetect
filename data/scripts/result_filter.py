import os


def filter_result(detect_result_path, cls, to_cls, x1_range: tuple):
    results = os.listdir(detect_result_path)
    for result_txt in results:
        f = open(os.path.join(detect_result_path, result_txt), "r+", encoding="utf-8")
        lines = f.readlines()
        cover_content = []

        if len(lines) == 1:
            line = lines[0]
            line_arr = line.split(" ")
            line_arr[0] = str(to_cls)
            line = str.join(" ", line_arr)
            x1 = line.split(" ")[1]
            if x1_range[1] > float(x1) > x1_range[0]:
                cover_content.append(line)
        else:
            clses = [line.split(" ")[0] for line in lines]

            # 不存在的时候，就全认为是指定类别。再根据范围筛选
            if not clses.__contains__(str(cls)):
                temp = []
                for line in lines:
                    line_arr = line.split(" ")
                    line_arr[0] = str(cls)
                    temp.append(str.join(" ", line_arr))
                lines = temp
                for i in range(len(clses)):
                    clses[i] = str(cls)

            if len(clses) > 0:
                for i in range(len(clses)):
                    if clses[i] == str(cls) and x1_range[0] < float(lines[i].split(" ")[1]) < x1_range[1]:
                        line_arr = lines[i].split(" ")
                        line_arr[0] = str(to_cls)
                        cover_content.append(str.join(" ", line_arr))
                        break
        if len(cover_content) > 0:
            f.seek(0)
            f.truncate(0)
            f.write(str(cover_content[0]))

        f.close()

        if len(cover_content) == 0:
            print("remove:", result_txt)
            os.remove(os.path.join(detect_result_path, result_txt))
            os.remove(os.path.join(detect_result_path, "../../images/train/" + result_txt.split(".")[0] + ".jpg"))


if __name__ == '__main__':
    filter_result(r"D:\MyRepo\AbandonedObjectDetect\datasets\my_dataset_1\labels\train", 24, 1, (0.37, 0.46))
