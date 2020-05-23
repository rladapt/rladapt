import random
import matplotlib.pyplot as plt
import os
import numpy as np
from utils.log import get_logger
from scipy import stats
from utils.config import logk_config
import subprocess
from .av_interface import get_size
import math
# (1) head(N) takes the top N entries
# (2) threshold(T) filters small entries whose count is smaller than T.
logger = get_logger(__name__)


class LogTask():
    def __init__(self, path, k_num, metrics, nums, bd_baseline):
        self.f = os.path.join(path, os.listdir(path)[nums])
        self.k_num = k_num
        self.metrics = metrics
        self.logger = logger
        self.bd_baseline = bd_baseline
        if self.f and os.path.isfile(self.f):
            self._open()
        self.operations = ['head_filter', 'frequency_filter']

    def apply(self, operation, parameter):
        if operation not in self.operations:
            raise Exception('operation not supported')
        if operation == 'head_filter':
            self._head = int(parameter)
        elif operation == 'frequency_filter':
            self._threshold = int(parameter)
        else:
            raise Exception('Should not appear')

    def run(self):
        self.group(self.metrics)
        topk1 = self.extract_topk()[:self.k_num]
        self.threshold(self._threshold)
        self.head(self._head)
        self._bandwidth = self.bandwidth()
        topk2 = self.extract_topk()[:self.k_num]
        if topk2.__len__() == 0:
            return -1
        topk2 += [topk2[-1]] * (len(topk1) - len(topk2))
        if max(topk2) == min(topk2):
            i = 0
            while topk1[i] == topk2[0]:
                i += 1
            topk2[i] = topk1[i]
        tau, p_value = stats.kendalltau(topk1, topk2, nan_policy='omit')
        if math.isnan(tau):
            print(topk2)
            raise Exception('tau is nan')
        tau = (tau + 1) / 2
        return tau

    def get_bandwidth(self):
        eps = 1e-6
        return max(self._bandwidth / logk_config['band_norm'], eps) / self.bd_baseline

    def _open(self):
        assert os.path.isfile(self.f)
        with open(self.f, 'r') as _f:
            self.c = _f.readlines()
        self.header = self.c[0]  # unsplited
        self.c = self.c[1:]  # unsplited
        self.header = self.header.strip().split(',')
        # self.logger.info(self.header)

    def init(self, header, content):
        self.header = self.header.strip().split(',')
        self.c = content

    def _split(self, i): return i.strip().split(',')[:-1]

    def group(self, metrics):
        assert metrics in self.header
        index = self.header.index(metrics)
        groups = {}
        for item in self.c:
            item_splited = self._split(item)
            if item_splited[index] in groups:
                groups[item_splited[index]].append(item)
            else:
                groups[item_splited[index]] = [item]
        self.groups = groups  # {metrics1:[item1, item2, ...], metrics2...}

    def show(self):
        assert self.groups
        x = np.arange(len(self.groups))
        labels = []
        y = []
        for key, val in sorted(self.groups.items(), key=lambda i: len(i[1]), reverse=True):
            labels.append(key)
            y.append(len(val))
        fig, ax = plt.subplots()
        x, y, labels = x[:10], y[:10], labels[:10]
        ax.bar(x, y)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        plt.show()

    def head(self, N):
        to_del = []
        cnt = 0
        for key, _ in sorted(self.groups.items(), key=lambda i: len(i[1]), reverse=True):
            cnt += 1
            if cnt > N:
                to_del.append(key)
        for key in to_del:
            del self.groups[key]

    def threshold(self, N):
        to_del = []
        for key, _ in self.groups.items():
            if self.groups[key].__len__() < N:
                to_del.append(key)
        for key in to_del:
            del self.groups[key]

    def bandwidth(self):
        to_str = ''
        for _, val in self.groups.items():
            to_str += ''.join(val)
        length = len(to_str.encode())
        mbps = length * 8 / 1e6
        self.logger.debug('bandwidth is {} mbps \n'.format(mbps))
        return mbps

    def extract_topk(self):
        return [key for key, _ in sorted(self.groups.items(), key=lambda i: len(i[1]), reverse=True)]


def logtask(path, train, metrics, head, threshold, k_num):
    task = LogTask(path, train)
    task.group(metrics)
    topk1 = task.extract_topk()[:k_num]
    task.threshold(threshold)
    task.head(head)
    bd = task.bandwidth()
    topk2 = task.extract_topk()[:k_num]
    if topk2.__len__() == 0:
        return bd, -1
    topk2 += [topk2[-1]] * (len(topk1) - len(topk2))
    if max(topk2) == min(topk2):
        i = 0
        while topk1[i] == topk2[0]:
            i += 1
        topk2[i] = topk1[i]
    tau, p_value = stats.kendalltau(topk1, topk2, nan_policy='omit')
    if math.isnan(tau):
        print(topk2)
        raise Exception('tau is nan')
    tau = (tau + 1) / 2
    eps = 1e-6
    return max(bd / logk_config['band_norm'], eps), tau


def makeslice():
    slice_size = 100 * 1e6 // 8  # 40Mbps
    slice_size = int(slice_size)
    f = open('/home/shen/Downloads/log.test.csv', 'r')
    root_name = '/home/shen/Downloads/slice.test'
    if os.path.isdir(root_name) == False:
        os.mkdir(root_name)
    subprocess.run("rm -rf /home/shen/Downloads/slice.test/*", shell=True)
    header = f.readlines(1)  # header
    c = True
    i = 0
    while c:
        c = f.readlines(slice_size)
        with open('{}/{:0>6d}.csv'.format(root_name, i), 'w') as newf:
            newf.write(header[0])
            for line in c:
                newf.write(line)
        i += 1


if __name__ == "__main__":
    # makeslice()
    # raise
    for i in os.listdir('/home/shen/Downloads/slice/')[:10]:
        print(logtask('/home/shen/Downloads/slice/' + i,
                      'train', 'accession', 10, 200, 30))
