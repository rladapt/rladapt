import socket
import struct
import time
import csv
import sys
from tqdm import tqdm
from threading import Thread, Lock

PORT = 8080
SERVER_HOST = '192.168.31.104'


class Server:
    def __init__(self, port=8080, host=socket.gethostname(), tcp=False):
        self.max_conn_num = 1000
        self.port = port
        self.host = host
        self.s = socket.socket()

        self.data_buffer = 0
        self.buffer_time = 0
        self.print_lock = Lock()
        self.rules = ''
        self.rules1 = ''
        self.rules_aws_pd = ''
        self.load_csv()
        self.log = []
        self.time_d = 0
        self.tcp = tcp

    def load_csv(self):
        self.rules_aws_pd = """0.04525424\t0.6356776089265633
        0.09673904\t0.7041041735360183
        0.12714928\t0.7550443880815195
        0.1465976\t0.7677538181384066
        0.17039951999999997\t0.7697994571990682
        0.2441192\t0.7706179185359575
        0.32822735999999997\t0.7980070248659696
        0.39711087999999994\t0.7997464783390894
        0.5777992\t0.8162972259501275
        0.6929367999999999\t0.8196899579527326
        1.2118958400000002\t0.8264638961940494
        1.4472084799999996\t0.8487621829573744
        1.67218416\t0.8555602020327604
        5.84679008\t0.8631580433001375
        7.912625440000001\t0.888554713291286
        9.67501248\t0.9039673324504633
        23.06764592\t0.9095936013489881"""

        self.rules1 = """1.7170205714285713\t0.620950849183538
        2.9355602285714286\t0.6826058521195076
        3.5013314285714285\t0.7379608251806474
        5.996930971428572\t0.8094561121431861
        7.826841600000001\t0.827844145359898
        10.6339648\t0.849639395193976
        11.962554742857142\t0.8538720026765533
        15.850294171428573\t0.8635661191882741
        19.49883428571429\t0.8708615991143901
        25.82987885714286\t0.8777951990048157"""

        self.rules = """1.11253424	0.847265634
        1.45200416	0.848931584
        1.67218416	0.855560202
        1.93794544	0.857230362
        2.26384192	0.861306249
        2.664224	0.866131699
        3.15864096	0.871189258
        3.76320672	0.871429232
        4.51185184	0.878114679
        5.44249744	0.883105158
        6.58902512	0.890194919
        7.99372864	0.895736535
        9.67501248	0.903967332
        11.62102608	0.909447055
        13.80159072	0.916312329
        16.17009312	0.922437244
        18.69413136	0.926849233
        21.40586992	0.933129304
        24.34141296	0.94047716
        27.53171808	0.944788642"""
        self.rules = [i.strip().split('\t') for i in self.rules_aws_pd.split('\n')]
        for i in range(len(self.rules)):
            self.rules[i][0] = float(self.rules[i][0])
            self.rules[i][1] = float(self.rules[i][1])

    def find_acc(self, width):
        if self.tcp:
            # TCP
            if width > 20:
                return 100.000
            else:
                return 0.0
        else:
            # others
            for i in reversed(self.rules):
                if width > i[0]:
                    return i[1]
            return 0

    def run(self):
        host = self.host
        port = self.port
        self.s.bind((host, port))

        self.s.listen(self.max_conn_num)
        print('server_started')
        while True:
            c, addr = self.s.accept()
            self.print_lock.acquire()
            print(addr, 'client connected')
            self.print_lock.release()
            # try to fix delay problem
            # time_before_send = time.time()
            # c.send(b'ack1')
            # time_delta = struct.unpack('>d', c.recv(8))[0]
            # time_rec = time.time()
            # time_delay = (time_rec - time_before_send) / 2
            # time_delta = time.time() - time_delta
            # self.time_d = time_delta - time_delay
            c.send(b'connected')
            t = Thread(target=self.main_loop_for_recv, args=(c, addr))
            t.start()

    def main_loop_for_recv(self, c, addr):
        while True:
            rec = c.recv(16)
            if not rec:
                continue
            try:
                msg_id = struct.unpack('>i', rec[:4])[0]
                send_time = struct.unpack('>d', rec[4:12])[0]
                data_size = struct.unpack('>i', rec[12:16])[0]
            except struct.error:
                continue

            if data_size < 0 or send_time < 0 or msg_id < 0 or msg_id > 10000000 or data_size > 10000000:
                continue
            t_send = time.time()
            c.send(b'ok1')
            data_all = c.recv(data_size)
            t_rec = time.time()

            if len(data_all) != data_size:
                left = c.recv(data_size - len(data_all))
                data_all += left

            # computing bandwidth
            self.data_buffer += data_size
            try:
                width = self.data_buffer / (t_rec - self.buffer_time) / 1024 / 1024 * 8
            except ZeroDivisionError:
                width = 0

            c.send(struct.pack('>d', int(width)))

            if t_rec - self.buffer_time >= 1:
                self.data_buffer = 0
                self.buffer_time = t_rec
            #  write csv
            delay = ((t_rec - t_send) / 2 * 1000)
            acc = self.find_acc(width)
            with open('result.csv', 'a+')as f:
                f_csv = csv.writer(f)
                f_csv.writerow([str(time.time()),
                                str(delay),
                                str(width),
                                str(acc)])
            # print
            if msg_id % 20 == 0:
                out_thing = ''.join(['id:%d  ' % msg_id,
                                     'delay:%.1fms  ' % delay,
                                     'width:%.4f Mbps  ' % width,
                                     'acc:%f  ' % self.find_acc(width),
                                     'size:%d/%d  ' % (len(data_all), data_size),
                                     'data:%s' % data_all[:10]])
                self.print_lock.acquire()
                print(out_thing)
                self.print_lock.release()
            if data_all == b'close':
                break
        self.print_lock.acquire()
        print(addr, 'client closed')
        self.print_lock.release()
        c.close()


if __name__ == '__main__':
    server = Server(PORT, SERVER_HOST, tcp=False)
    server.run()
