import cmath


def second2str(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


class BufferList(list):
    """
    只存储一定长度的数据，但是记录存过多少数据。
    """

    def __init__(self, seq=(), max_length=500):
        super(BufferList, self).__init__(seq)
        self.offset = 0

    def min_index(self):
        return self.offset

    def max_index(self):
        return self.offset + len(self) - 1

    def __getitem__(self, item):
        return super(BufferList, self).__getitem__(max(0, item - self.offset))

    def append(self, __object) -> None:
        super(BufferList, self).append(__object)
        while len(self) > 500:
            self.pop()

    def pop(self, **kwargs):
        self.offset += 1
        super(BufferList, self).pop(0)

    def clear(self) -> None:
        self.offset = 0
        super(BufferList, self).clear()


class LimitQueue(list):
    """
    只能存储一定长度数据的，会自动弹出的queue
    """

    def __init__(self, max_length):
        super(LimitQueue, self).__init__()
        self.max_length = max_length

    def append(self, __object):
        super(LimitQueue, self).append(__object)
        while self.__len__() > self.max_length:
            self.pop(0)

    def is_full(self):
        return self.__len__() >= self.max_length

    def set_maxlength(self, max_length):
        self.max_length = max_length


def get_point_center_distance(p1, p2):
    """
    获取两个点中心的距离
    :param p1: [x1,y1,x2,y2...]
    :param p2: [x1,y1,x2,y2...]
    :return:
    """
    center1 = (p1[0] + p1[2]) / 2, (p1[1] + p1[3]) / 2
    center2 = (p2[0] + p2[2]) / 2, (p2[1] + p2[3]) / 2
    x_distance = abs(center1[0] - center2[0])
    y_distance = abs(center1[1] - center2[1])
    return cmath.sqrt(x_distance ^ 2 + y_distance ^ 2)


def get_max_distance_in_points(points):
    pass
