from requests import get, post
import time
import numpy as np
from utils.search import find_le
import os
import copy


class runtime():
    def __init__(self, config):
        self.cur = 15
        self.low = 0
        self.high = 20
        self.min_size = 0.1
        self.min_time = 0.1
        self.eps = 1
        self.times = {}
        self.add_config(config)
        if hasattr(self, 'log_path'):
            self.log = open(self.log_path, 'a')
        else:
            self.log = open('./log.csv', 'a')
        self.log.write('time bandwidth accuracy latency\n')

        # profile csv
        self.model = [[], []]
        with open('rl-adapt.csv', 'r') as f:
            f.readlines(1)  # header
            while True:
                bd, f1score = f.readlines(1)[0].strip().split(' ')
                bd, f1score = float(bd), float(f1score)
                if bd < 22:
                    self.model[0].append(bd)
                    self.model[1].append(f1score)
                else:
                    break

        # grace period
        self.grace_times = []

    def add_config(self, config):
        for key, val in config.items():
            setattr(self, key, val)
            print('set {} to {}'.format(key, val))

    def onesec(self, size):
        size = int(size)  # bytes
        url = 'http://192.168.50.29:8888/{}'.format(size)
        start = time.time()
        response = get(url)
        t = time.time() - start
        print('duration: {}s, speed {}Mbps, internal {} Mbps '.format(
            t, size/1e6/t, self.cur))
        return t  # bytes per second

    def predict(self, bandwidth):
        i = find_le(self.model[0], bandwidth)
        return self.model[0][i], self.model[1][i]

    @property
    def now(self):
        return time.time() - self.times['start']

    @property
    def scaled_now(self):
        return self.now - self.grace_time

    @property
    def latency(self):
        return self.now - self.times['count'] * (self.min_time) - self.grace_time

    @property
    def grace_time(self):
        return sum(self.grace_times)

    def scaled_minsize(self, latency):
        return min(1, self.min_size * (1 + 5 * abs(latency - self.min_time) / min(latency, self.min_time)))

    def run(self, length):
        self.times['start'] = time.time()
        self.times['count'] = 0
        while self.scaled_now < length:
            cur, accuracy = self.predict(self.cur)
            latency = self.onesec(cur * 1e6 * self.min_time)
            self.times['count'] += 1
            self.grace_times.append(max(0, latency - self.min_time))
            if latency - self.min_time < self.eps:
                print('[UP] latency: {:.5f}'.format(latency - self.min_time))
                self.cur = np.clip(self.cur + self.scaled_minsize(latency),
                                   self.low, self.high)
            else:
                print('[DWON] latency: {:.5f}'.format(self.latency))
                self.cur = np.clip(self.cur - self.scaled_minsize(latency),
                                   self.low, self.high)
            self.log.write('{:.5f} {:.5f} {:.5f} {:.5f}\n'.format(
                self.scaled_now,
                self.cur * self.min_time / latency,
                accuracy,
                max(0, self.latency)
            ))
            self.log.flush()
            time.sleep(max(0, -self.latency))

    def __del__(self):
        self.log.close()


def splitlog2(path):
    # split min, max, medium
    with open(path, 'r') as f:
        headers = f.readlines(1)[0].strip().split(' ')
        contents = f.readlines()
        contents = [i.strip().split(' ') for i in contents]
        contents = np.array(contents).astype(np.float64)
        result = {'min': [], 'max': [], 'mid': []}
        _tmp = {}
        for i in contents:
            # [time, metrics]
            if int(i[0]) not in _tmp:
                _tmp[int(i[0])] = [i[1]]
            else:
                _tmp[int(i[0])].append(i[1])
        for key, val in _tmp.items():
            if len(val) >= 5:
                val = sorted(val)
                val = val[int(len(val)*0.025):][:min(-1, -int(len(val)*0.075))]
            _min, _max, _mid = min(val), max(val), np.median(val)
            if _min == 0:
                _min = 1e-3
            result['min'].append([key, _min])
            result['max'].append([key, _max])
            result['mid'].append([key, _mid])
        for key, val in result.items():
            result[key] = sorted(val, key=lambda i: i[0])
        for suffix in result:
            with open(path[:-4] + '-{}.csv'.format(suffix), 'w') as f2:
                f2.write('{} {}\n'.format(headers[0], headers[1]))
                for i in result[suffix]:
                    f2.write('{} {}\n'.format(i[0], i[1]))


def splitlog(path='log.csv'):
    with open(path, 'r') as f:
        headers = f.readlines(1)[0].strip().split(' ')
        contents = f.readlines()
        contents = [i.strip().split(' ') for i in contents if 'time' not in i]
        contents = np.array(contents)
        to_change = []
        to_remove = []
        for i in headers[1:]:
            filename = '{}-{}.csv'.format(headers[0], i)
            with open(filename, 'w') as f2:
                f2.write('{} {}\n'.format(headers[0], i))
                x = contents.T[0]
                y = contents.T[headers.index(i)]
                for index in range(len(x)):
                    if i == 'bandwidth' and float(y[index]) > 30:
                        to_remove.append(index)
                    if i == 'accuracy' and float(y[index]) == 0:
                        y[index] = y[index-1]
                    if i == 'bandwidth' and index < len(x) - 1 and float(y[index+1]) == 0:
                        y[index+1] = 1
                    if i == 'bandwidth' and index > 0 and index < len(x) - 1 and float(y[index])/float(y[index-1]) > 2.5 and float(y[index])/float(y[index+1]) > 2.5:
                        to_change.append(index)
                    if index in to_remove:
                        continue
                    if index in to_change:
                        y[index] = y[index - 1]
                    f2.write('{} {}\n'.format(x[index], y[index]))
            splitlog2(filename)


if __name__ == "__main__":
    r = runtime({})
    r.run(204)
    splitlog()
