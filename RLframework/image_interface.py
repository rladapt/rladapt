from utils.config import pd_config
import uuid
import subprocess
import os
import random
import cv2
from multiprocessing import Pool, cpu_count
from utils import iou, stat, log
import re
import shutil
import logging
from av_interface import *
logger = log.get_logger('visiontask')


class vision():
    def __init__(self, path, nums, skip, res, quantizer, debug, bd_baseline):
        self.skip = skip
        self.width = 0
        self.height = 0
        self.id = uuid.uuid1().hex
        self.path = path
        self.resized_path = os.path.join(path, 'resized' + self.id)
        if not os.path.exists(self.resized_path):
            os.mkdir(self.resized_path)
        self.decoded_path = os.path.join(os.getcwd(), 'tmp' + self.id)
        if not os.path.exists(self.decoded_path):
            os.mkdir(self.decoded_path)
        self.video_path = os.path.join(os.getcwd(), self.id + 'output2.mp4')
        self.quantizer = quantizer
        self.nums = nums
        self.bd = 0
        self.res = res
        self.bd_baseline = bd_baseline
        self.debug = debug

    def h264encode(self):
        encode_jpg(self.resized_path, self.video_path,
                   self.quantizer, 30 // self.skip, self.nums[0])

    def h264decode(self):
        decode_jpg(self.video_path, self.decoded_path)
        cur = os.listdir(self.decoded_path).__len__()
        indexs = [1 + self.skip * i for i in range(cur)]
        for index in indexs[::-1]:
            shutil.move(os.path.join(self.decoded_path, "{:0>6d}.jpg".format(cur)),
                        os.path.join(self.decoded_path, "{:0>6d}.jpg".format(index)))
            cur -= 1

    def run_hog(self, baseline=None):
        os.environ['INPUT'] = self.decoded_path
        os.environ['NUMS'] = ','.join([str(i+1)
                                       for i in range(self.nums.__len__())])
        # os.environ['NUMS'] = ','.join([str(i) for i in self.nums])
        os.environ['SKIP'] = str(self.skip)
        self.result = subprocess.check_output(['/home/shen/research/RL/WANStream/src/pd'],
                                              universal_newlines=True).strip()
        if baseline:
            with open(baseline, 'w', encoding='utf-8') as f:
                f.write(self.result)
        self.result = stat.FrameStat(self.result)
        self._set_skip()
        return self.result

    def _set_skip(self):
        if self.skip == 0:
            return
        skip_nums = []
        for i in range(self.nums.__len__()):
            if i % self.skip != 0:
                skip_nums.append(i)
        self.result.set_skip(skip_nums)

    def resize(self, parallel=False):
        resy = self._adjust_res()
        if parallel:
            p = Pool(cpu_count())
            for index, i in enumerate(self.nums):
                if index % self.skip == 0:
                    p.apply_async(
                        self._resize, (int(i), self.path, self.res, resy))
            p.close()
            p.join()
        else:
            for index, i in enumerate(self.nums):
                if index % self.skip == 0:
                    self._resize(int(i), self.path, self.res, resy)

    def get_IoU(self, baseline_path=None):
        if baseline_path == None:
            return -1
        with open(baseline_path, 'r', encoding='utf-8') as f:
            self.baseline = stat.FrameStat(
                f.read())[int(self.nums[0])-1:int(self.nums[-1])]
        self.result = self.result[:]
        assert(len(self.baseline) == len(self.result))

        sumacc = []
        for i, frame_baseline in enumerate(self.baseline):

            IoU = 0
            for _, frame_baseline_per in enumerate(frame_baseline):
                IoU += len([_ for frame_result in self.result[i] if
                            self._IoU(frame_baseline_per, frame_result) >= 0.5])
            tp = min(IoU, len(frame_baseline))
            tp_and_fp = self.result[i].__len__()
            tp_and_fn = frame_baseline.__len__()
            fp = tp_and_fp - tp
            fn = tp_and_fn - tp
            f1score = stat.Stat(tp, fp, fn).f1score()
            # logger.info(f1score, self.nums[i], i, '\n',self.result[i], '\n', self.baseline[i])
            if tp == 0 and fp == 0 and fn == 0:
                sumacc.append(1)
            else:
                sumacc.append(f1score)
        logger.info('avg acc is {}'.format(sum(sumacc)/len(sumacc)))
        return sum(sumacc)/len(sumacc)

    def _adjust_res(self):
        img0 = cv2.imread(
            '{}/{:0>6d}.jpg'.format(self.path, int(self.nums[0])))
        # ffmpeg does not accept (width % 2 != 0)
        height0, width0, _ = img0.shape
        if int(self.res * width0 + 0.5) % 2:
            self.res = int(self.res * width0 + 0.5) // 2 * 2 / width0
        resy = self.res
        if int(resy * height0 + 0.5) % 2:
            resy = int(resy * height0 + 0.5) // 2 * 2 / height0
        assert(self.width == 0 and self.height == 0)
        assert(self.res > 0 and self.res <= 1)
        self.width = int(width0 * self.res + 0.5)
        self.height = int(height0 * resy + 0.5)
        assert(self.width % 2 == 0 and self.height % 2 == 0)
        if self.debug:
            logger.info("width: {}, height: {}".format(
                self.width, self.height))
        return resy

    def _resize(self, i, path, resolutionx, resolutiony):
        ipath = '{}/{:0>6d}.jpg'.format(path, i)
        resized_ipath = '{}/{:0>6d}.jpg'.format(self.resized_path, i)
        img = cv2.imread(ipath)
        resized = cv2.resize(img, None, fx=resolutionx,
                             fy=resolutiony, interpolation=cv2.INTER_AREA)
        cv2.imwrite(resized_ipath, resized)

    def _IoU(self, img0, img1):
        # calculate IoU of a given set of nums, return Positive if IoU larger than 0.5
        # the baseline is imgs without any degradation
        # -------------------------------------------
        # img0, img1 only contains 4 floats [x,y,h,w]
        # img0 means img without degradation
        # resolution ranges from (-1, 0]
        def init(img): return [float(img[0]), float(img[1]),
                               float(img[0]) + float(img[2]), float(img[1]) + float(img[3])]

        img0 = init(img0)
        img1 = init(img1)
        return iou.get_IoU(img0, img1)

    def bandwidth(self):
        self.bd = get_size(self.video_path) / 1e6 * 8
        if self.bd_baseline:
            logger.info('bandwidth is {} Mbps, relative: {}'.format(
                self.bd / len(self.nums) * 30, self.bd/len(self.nums)*30/self.bd_baseline))
        return self.bd / len(self.nums) * 30

    def clean(self):
        if self.debug:
            return
        # logger.info('del {} {} {}'.format(self.decoded_path,
        #                                  self.resized_path, self.video_path))
        subprocess.run('rm -rf {} {} {}'.format(self.decoded_path,
                                                self.resized_path, self.video_path), shell=True)

    def __del__(self):
        self.clean()


TRAIN_NUM = 1051
TEST_NUM = 1501
TRAIN_PATH = '/home/shen/research/awstream-data/train/MOT16-04/img1'
TEST_PATH = '/home/shen/research/awstream-data/test/MOT16-03/img1'
TRAIN_BASELINE = '/home/shen/research/RL/WANStream/profile/baseline.txt'
TEST_BASELINE = '/home/shen/research/RL/WANStream/profile/baseline_test.txt'
TRAIN_BANDWIDTH = pd_config['band_norm']
TEST_BANDWIDTH = pd_config['band_norm']
BATCH_SIZE = 60


def visiontask2(resolution, batch_size, skip, mode, quantizer, debug=False):
    logger.info("res = {}, batch = {}, skip = {}, quant = {} mode = {}".format(
        resolution, batch_size, skip, quantizer,  mode))
    if skip == 0:
        skip = 1
    assert(resolution > 0 and resolution <= 1)
    assert(skip)
    if mode == 'train':
        start = random.randint(1, TRAIN_NUM - BATCH_SIZE)
        path = TRAIN_PATH
        baseline = TRAIN_BASELINE
        bandwidth = TRAIN_BANDWIDTH
    else:
        assert(mode == 'test')
        start = random.randint(1, TEST_NUM - BATCH_SIZE)
        path = TEST_PATH
        baseline = TEST_BASELINE
        bandwidth = TEST_BANDWIDTH
    if isinstance(batch_size, (tuple, list)):
        nums = [str(i) for i in range(batch_size[0], batch_size[1])]
    else:
        nums = [str(i) for i in range(start, start+batch_size)]

    v = vision(path, nums, skip, resolution, quantizer, debug, bandwidth)
    try:
        v.resize()
        v.h264encode()
        v.h264decode()
        v.run_hog()
        f1score = v.get_IoU(baseline)
        bd = v.bandwidth()
        v.clean()
        return bd / bandwidth, f1score
    except Exception as e:
        v.clean()
        raise e


if __name__ == '__main__':
    for _ in range(1):
        logger.info(visiontask2(1, 60, 1, 'train', 0, debug=False))
