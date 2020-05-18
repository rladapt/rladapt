import numpy as np
import random


class shaper():
    def __init__(self, path):
        self.f = open(path, 'w')

    def __del__(self):
        self.f.close()

    def end(self):
        self.f.write('main')
        self.f.close()

    def initmain(self, bandwidth, time):
        assert len(bandwidth) == len(time)
        self.f.write('function main(){\n')
        self.f.write('\techo "start shaping"\n')
        self.f.write('\ttrap "stop" SIGHUP SIGINT SIGTERM\n')
        self.f.write(
            '\tsudo tc qdisc add dev lo root handle 1: htb default 12\n')
        self.f.write(
            '\tsudo tc filter add dev lo protocol ip parent 1: prio 1 u32 match ip dst 127.0.0.1 flowid 1:12\n')
        for i in range(len(bandwidth)):
            if i == 0:
                key = 'add'
            else:
                key = 'change'
            self.f.write('\tsudo tc class {} dev lo parent 1:1 classid 1:12 htb rate {}mbit ceil {}mbit\n'.format(
                key, bandwidth[i], bandwidth[i]))
            if i == 0:
                self.f.write(
                    '\t sudo tc qdisc add dev lo parent 1:12 handle 12:  netem delay 50ms 10ms distribution normal\n'
                )
            self.f.write('\t echo "set to {} mbit"\n'.format(bandwidth[i]))
            self.f.write('\tsleep {}\n\n'.format(time[i]))
        self.f.write('\tstop\n')
        self.f.write('}\n')

    def initstop(self):
        self.f.write('function stop(){\n')
        self.f.write('\techo "stopped" \n')
        self.f.write('\tsudo tc qdisc del dev lo root\n')
        self.f.write('\t exit 0 \n')
        self.f.write('}\n')


class FakeRandom():
    def __init__(self, low, high):
        self.high = high
        self.start = random.randint(low, self.high)
        self.low = low
        self.cur = self.start

    @property
    def randint(self):
        tmp = self.cur
        self.cur = (self.cur + 1) % self.high
        if self.cur == self.start:
            self.start = random.randint(0, self.high)
            self.cur = self.start
        return tmp


class RandNet():
    def __init__(self, low, high):
        self.rand = FakeRandom(low, high)
        self.low = low
        self.high = high

    @property
    def cur(self):
        return np.clip(np.random.normal(self.rand.randint, 0.5), self.low, self.high)


if __name__ == "__main__":
    s = shaper('/home/shen/research/RL/WANStream/runtime/traffic_control/main.sh')
    net = RandNet(2, 20)
    # bds = [net.cur for i in range(20)]
    # random.shuffle(bds)
    bds = [20, 18, 16, 14, 12, 10, 8, 6, 4,
           2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    sleep = [10 for i in range(19)]
    sleep[bds.index(2)] = 20
    s.initmain(bds, sleep)
    s.initstop()
    s.end()
