from image_interface import vision
from utils.darknet import detect, load_net
import contextlib
import io
import subprocess
from utils import stat, log
import copy
import random
import cv2
import shutil
from av_interface import *
from multiprocessing import Pool, cpu_count, set_start_method, freeze_support
import os
from utils.config import ar_config
from threading import Lock
import uuid


class ARtask(vision):
    def __init__(self, path, nums, skip, res, quantizer, debug, bd_baseline=None):
        super().__init__(path, nums, skip,
                         res, quantizer, debug, bd_baseline)
        self.draw_path = os.path.join(path, 'draw' + uuid.uuid1().hex)
        self.text_path = self.draw_path+'text'
        if not os.path.exists(self.draw_path):
            os.mkdir(self.draw_path)
        self.logger = log.get_logger(__name__)

    def add_net(self, net):
        self.net = net

    def _IoU(self, img0, img1):
        # if img0[-1] != img1[-1]:
        #    return 0
        return super()._IoU(img0, img1)

    def _set_skip(self):
        diff = len(self.nums) - len(self.result)
        last = self.result[len(self.result)-1]
        for _ in range(diff):
            self.result.append(copy.deepcopy(last))
        super()._set_skip()

    def run_detect(self, baseline=None):
        detect(self.net, self.decoded_path, self.draw_path)
        with open(self.text_path, 'r', encoding='utf-8') as f:
            self.result = f.read()
        if baseline and os.path.exists(baseline) == False:
            with open(baseline, 'w', encoding='utf-8') as f:
                f.write(self.result)
        # for skip, add skipped imgs to self.result
        self.result = stat.FrameStat(self.result)
        self._set_skip()
        self.result = self.result[:]

    def clean(self):
        if self.debug:
            return
        subprocess.run('rm -rf {} {} {} {} {}'.format(self.decoded_path,
                                                      self.resized_path, self.video_path, self.draw_path, self.text_path), shell=True)


def artask(net, resolution, batch_size, skip, mode, quantizer, debug=False):
    print("[RES] {}, [SKIP] {}, [QUANT] {}".format(
        resolution, skip, quantizer))
    TRAIN_PATH = '/home/shen/Downloads/darknet/video/imgs_small'
    TRAIN_BASELINE = os.path.join(TRAIN_PATH, 'baseline.txt')
    TRAIN_NUM = [i for i in os.listdir(TRAIN_PATH) if '.jpg' in i].__len__()

    TEST_PATH = '/home/shen/Downloads/darknet/video/imgs_test'
    TEST_BSAELINE = os.path.join(TEST_PATH, 'baseline.txt')
    if mode == 'test':
        assert isinstance(batch_size, (tuple, list))
        path = TEST_PATH
        baseline = TEST_BSAELINE
    if mode == 'train':
        path = TRAIN_PATH
        baseline = TRAIN_BASELINE

    if isinstance(batch_size, (tuple, list)):
        nums = [str(i) for i in range(batch_size[0], batch_size[1])]
    else:
        start = random.randint(1, TRAIN_NUM - batch_size)
        nums = [str(i) for i in range(start, start+batch_size)]
        if debug:
            print("batch: start {} end{}".format(start, start+batch_size))
    task = ARtask(path, nums, skip, resolution,
                  quantizer, debug, ar_config['band_norm'])
    task.add_net(net)
    try:
        task.resize(False)
        task.h264encode()
        task.h264decode()
        task.run_detect(baseline)
        f1socre = task.get_IoU(baseline)
        bd = task.bandwidth()
        task.clean()
        return bd / ar_config['band_norm'], f1socre
    except Exception as e:
        print(e)


def reduce(result):
    print("bd {}, f1socre {}".format(result[0], result[1]))


class net():
    def __init__(self):
        self._net = []

    @property
    def net(self):
        if self._net == []:
            self._net = load_net(b"/home/shen/Downloads/darknet/cfg/yolov3-tiny.cfg",
                                 b"/home/shen/Downloads/darknet/weights/yolov3-tiny.weights", 0)
        return self._net


net = net()


if __name__ == "__main__":
    _net = load_net(b"/home/shen/Downloads/darknet/cfg/yolov3-tiny.cfg",
                    b"/home/shen/Downloads/darknet/weights/yolov3-tiny.weights", 0)

    print("bd {}, f1socre {}".format(
        *artask(_net, 1, (1, 501), 1, 'train', 0, True)))


# batch: start 1072 end1102
