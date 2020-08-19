from apis.adaptmodel import AdaptModel
from lib.image_interface import visiontask2 as visiontask
from lib.image_interface import vision
from lib.evaluate_interface import bandwidth_reward, accuracy_reward
import numpy as np
import time
from utils.config import pd_config
import os
import random

TRAIN_NUM = 1050
BATCH_SIZE = 30
TEST_NUM = 1500
TRAIN_DATA = '../dataset/pd'
TEST_DATA = '../dataset/pd_test'


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
    dataset = '../dataset/pd'
    ground_truth = os.path.join(dataset, 'ground_truth.txt')
    start = random.randint(1, TRAIN_NUM + 1 - BATCH_SIZE)
    stream = vision(dataset, np.arange(start, start + BATCH_SIZE),
                    ground_truth, pd_config.get('band_norm', 140))
    '''
        Start applying degradation to the stream:
        * apply down_smaple
        * apply adjust_frame
        * apply adjust_quality
    '''
    stream.apply('down_sample', resolution)
    stream.apply('adjust_frame', frame_rate)
    stream.apply('adjust_quality', encode_quality)

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
    dataset = TEST_DATA
    ground_truth = os.path.join(dataset, 'ground_truth.txt')
    start = 1
    stream = vision(dataset, np.arange(start, start + TEST_NUM),
                    ground_truth, pd_config.get('band_norm', 140))
    '''
        Start applying degradation to the stream:
        * apply down_smaple
        * apply adjust_frame
        * apply adjust_quality
    '''
    stream.apply('down_sample', resolution)
    stream.apply('adjust_frame', frame_rate)
    stream.apply('adjust_quality', encode_quality)

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
    total = - np.sqrt(b_reward ** 2 + a_reward ** 2)
    print("environment {}, bandwidth {}, f1socre {}, reward {}".format(
        environment * pd_config.get('band_norm', 140), bandwidth * pd_config.get('band_norm', 140), f1score, total))
    # print("b_reward {}, a_reward {}".format(b_reward, a_reward))
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
        **pd_config,
        'degrade': degrade_func,
        'evaluate': evaluate_func,
        'use_sac': True
    }
    # [1,2,3,5,6,10,15,30]
    a = AdaptModel(0, 40)\
        .add('res', (0.15, 1), 'continuous')\
        .add('skip', 7, 'discrete')\
        .add('quant', 50, 'discrete')\
        .add_degrade(degrade_func) \
        .add_evaluate(evaluate_func)\
        .add_profile('../models/pd') \
        .build()

    # a.test(40)  # if test
    a.train(0)  # if train
