from utils.config import ar_config
from lib.ar_interface import net, ARtask
from lib.evaluate_interface import bandwidth_reward, accuracy_reward
from utils.darknet import load_net
from utils.log import get_logger
from apis.adaptmodel import AdaptModel
import numpy as np
import os
import random


TRAIN_NUM = 500
BATCH_SIZE = 30
TEST_DATA = '../dataset/ar_test'
TEST_START = 501
TEST_NUM = 200  # 550


def degrade_func(resolution, frame_rate, encode_quality):
    frame_rate = [1, 2, 3, 5, 6, 10, 15, 30][int(frame_rate)]
    encode_quality = np.arange(0, 51, 1)[int(encode_quality)]
    print('[RESOLUTION]: {},[FRAME_RATE]: {}, [ENCODE_QUALITY]: {}'.format(
        resolution, frame_rate, encode_quality
    ))

    ''' 
        Read imgs as arrays:

        * First, point where is the `dataset` and 
            where `ground_truth` should be stored.
        * Second, decide the where the `BATCH_SIZE` should start from.
        * Create the stream.
    '''
    dataset = '../dataset/ar_dev'
    ground_truth = os.path.join(dataset, 'ground_truth.txt')
    start = random.randint(1, TRAIN_NUM + 1 - BATCH_SIZE)
    stream = ARtask(dataset, np.arange(start, start + BATCH_SIZE),
                    ground_truth, ar_config.get('band_norm', 160), net.net)
    # stream = ARtask(dataset, np.arange(1, 1 + 1050),
    #                 ground_truth, ar_config.get('band_norm', 160), net.net)

    '''
        Start applying degradation to the stream:
        * apply down_smaple
        * apply adjust_frame
        * apply adjust_quality
    '''
    stream.apply('down_sample', resolution)
    stream.apply('adjust_frame', frame_rate)
    stream.apply('adjust_quality', encode_quality)

    # stream.apply('down_sample', 1)
    # stream.apply('adjust_frame', 1)
    # stream.apply('adjust_quality', 0)

    '''
        Complete operations and 
        get value of reward metrics from degraded stream.
        * run specific application through degraded data
            and get accuracy
        * get bandwidth from degraded stream
    '''
    accuracy = stream.run()
    bandwidth = stream.get_bandwidth()

    return bandwidth, accuracy


def degrade_func_test(resolution, frame_rate, encode_quality):
    frame_rate = [1, 2, 3, 5, 6, 10, 15, 30][int(frame_rate)]
    encode_quality = np.arange(0, 51, 1)[int(encode_quality)]
    print('[RESOLUTION]: {},[FRAME_RATE]: {}, [ENCODE_QUALITY]: {}'.format(
        resolution, frame_rate, encode_quality
    ))

    ''' 
        Read imgs as arrays:

        * First, point where is the `dataset` and 
            where `ground_truth` should be stored.
        * Second, decide the where the `BATCH_SIZE` should start from.
        * Create the stream.
    '''
    dataset = '../dataset/ar_dev'
    ground_truth = os.path.join(dataset, 'ground_truth.txt')
    start = TEST_START
    stream = ARtask(dataset, np.arange(start, start + TEST_NUM),
                    ground_truth, ar_config.get('band_norm', 160), net.net)
    # stream = ARtask(dataset, np.arange(1, 1 + 1050),
    #                 ground_truth, ar_config.get('band_norm', 160), net.net)

    '''
        Start applying degradation to the stream:
        * apply down_smaple
        * apply adjust_frame
        * apply adjust_quality
    '''
    stream.apply('down_sample', resolution)
    stream.apply('adjust_frame', frame_rate)
    stream.apply('adjust_quality', encode_quality)

    # stream.apply('down_sample', 1)
    # stream.apply('adjust_frame', 1)
    # stream.apply('adjust_quality', 0)

    '''
        Complete operations and 
        get value of reward metrics from degraded stream.
        * run specific application through degraded data
            and get accuracy
        * get bandwidth from degraded stream
    '''
    accuracy = stream.run()
    bandwidth = stream.get_bandwidth()

    return bandwidth, accuracy


def evaluate_func(environment, bandwidth, f1score):
    b_reward = bandwidth_reward(environment, bandwidth)
    a_reward = accuracy_reward(f1score)
    total = 0.5 * b_reward + a_reward
    print("environment {}, bandwidth {}, reward {}".format(
        environment * ar_config.get('band_norm', 140), bandwidth * ar_config.get('band_norm', 140), b_reward * 2 + a_reward))
    print("b_reward {}, a_reward {}".format(b_reward, a_reward))
    return total


if __name__ == "__main__":
    '''
        if first run

    _TMP_BATCH_SIZE = BATCH_SIZE
    BATCH_SIZE = TRAIN_NUM
    degrade_func(1, 1, 0)
    BATCH_SIZE = _TMP_BATCH_SIZE
    '''

    config = {
        'distill_mode': 1
    }
    model = AdaptModel(0, 40) \
        .add('res', (0.15, 1), 'continuous')\
        .add('skip', 7, 'discrete')\
        .add('quant', 50, 'discrete')\
        .add_degrade(degrade_func)\
        .add_evaluate(evaluate_func)\
        .add_profile('../models/ar') \
        .add_config(config) \
        .build()
    # degrade_func(1, 0, 0)
    model.train(0)
