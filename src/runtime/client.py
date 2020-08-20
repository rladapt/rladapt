import socket
import time
import struct
import requests
from multiprocessing import Pool
from threading import Thread

PORT = 8080
SERVER_HOST = '192.168.31.104'
NET_URL = 'http://localhost:8888/'


class Client:
    def __init__(self, port=8080, server_host=socket.gethostname(), tcp=False):
        self.port = port
        self.host = server_host
        self.s = socket.socket()
        self.id = 0
        self.width = 40
        self.tcp = tcp

    def send_data(self, data_str):
        """
        > big end mode
        i int 4 bytes
        d double 8 bytes
        """
        self.id += 1
        data = data_str.encode()
        t_send = time.time()
        # make header
        head = struct.pack('>i', self.id) + struct.pack('>d', t_send) + struct.pack('>i', len(data))
        self.s.send(head)
        while True:
            # wait for first tag
            ret1 = self.s.recv(3)
            if ret1:
                break
        if ret1 == b'er1':
            print('failed', self.id, t_send, len(data))
            return
        self.s.sendall(data)
        print(self.id, t_send, len(data), data[:30])
        while True:
            # wait for second tag
            ret2 = self.s.recv(8)
            if ret2:
                self.width = struct.unpack('>d', ret2)[0]
                break

    def close(self):
        self.s.close()

    def run(self):
        host = self.host
        port = self.port
        self.s.connect((host, port))
        print(self.s.recv(1024))
        self.send_data('start')
        for _ in range(50000):
            if self.tcp:
                self.send_data('test_data_fa' * 500)
            else:
                if self.width > 20:
                    handle = 20
                else:
                    handle = int(self.width)
                self.send_data('test_data_fa' * (100 + 15 * handle))

        self.send_data('close')
        self.close()


def new_thread():
    client = Client(PORT, SERVER_HOST, tcp=False)
    client.run()


def request_net():
    while True:
        requests.get('http://localhost:8888/', params={'bd': 99})
        time.sleep(0.03)


if __name__ == '__main__':
    # start net request process
    p = Pool(1)
    p.apply_async(request_net)

    # start 100 client threads
    ts = []
    for i in range(100):
        t = Thread(target=new_thread)
        t.start()
        ts.append(t)
    for i in ts:
        i.join()

    p.close()
    p.join()
