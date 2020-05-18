
import re
import copy


class Stat():
    def __init__(self, tp, fp, fn):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        # print("tp:{}, fp:{}, fn:{}".format(tp, fp, fn))

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def f1score(self):
        if self.tp == 0:
            return 0
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())


class Frame():
    def __init__(self, frameraw, skip=False):
        self.raw = copy.deepcopy(frameraw)
        frameraw = frameraw.split('\n')
        self.header = frameraw[0][-10:]
        def str2float(l): return [float(i) for i in l]
        self.records = [str2float(i.split(',')[-4:]) + [i.split(',')[2].strip()]
                        for i in frameraw[1:] if i]
        if self.records.__len__() == 1 and self.records[0] == [0, 0, 0, 0]:
            self.no_detection = True
        else:
            self.no_detection = False
        self.skip = skip

    def debug(self):
        print("no_detection: {}".format(self.no_detection))

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return self.records.__len__()

    def __repr__(self):
        return self.header + '\n' + str(self.records) + '\n'

    def __lt__(self, frame):
        return self.records < frame.records


class FrameStat():
    def __init__(self, framesraw):
        frames = re.split('new_frame\n', framesraw)[1:]
        self.frames = [Frame(i) for i in frames]
        self.frames[0].no_detection = False

    def set_skip(self, nums) -> None:
        for i in nums:
            self.frames[i].skip = True

    def append(self, frame):
        if isinstance(frame, Frame) == False:
            raise Exception("append none-Frame instance!")
        self.frames.append(frame)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        if key < 0 or key >= self.frames.__len__():
            raise KeyError("index out of range")

        # return the last detected frame
        while self.frames[key].skip == True:
            key -= 1

        return self.frames[key]

    def __len__(self):
        return len(self.frames)

    def __repr__(self):
        return self.frames.__repr__()
