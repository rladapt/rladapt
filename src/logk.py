from apis.adaptmodel import AdaptModel, AWStream
from lib.distlogk_interface import LogTask
from lib.evaluate_interface import bandwidth_reward, accuracy_reward
import os
import time
import random
from utils.config import logk_config
from utils.log import get_logger
import math
import numpy as np

K_NUM = 30
TRAIN_NUM = 122


def degrade_func(head, threshold):
    time.sleep(1)
    print('[HEAD]: {},[THRESHOLD]: {}'.format(
        head, threshold
    ))
    ''' 
        Read imgs as arrays:

        * First, point where is the `dataset` and 
            where `ground_truth` should be stored.
        * Second, decide the where the `BATCH_SIZE` should start from.
        * Create the stream.
    '''
    dataset = '../dataset/logk'
    start = random.randint(1, TRAIN_NUM + 1)
    stream = LogTask(dataset, K_NUM, 'accession', start,
                     logk_config.get('band_norm', 100))

    '''
        Start applying degradation to the stream:
        * apply down_smaple
        * apply adjust_frame
        * apply adjust_quality
    '''
    stream.apply('head_filter', head)
    stream.apply('frequency_filter', threshold)

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


def evaluate_func(environment, bandwidth, tau):
    b_reward = bandwidth_reward(environment, bandwidth)
    a_reward = accuracy_reward(tau)
    print("environment {}, bandwidth {}, reward {}".format(
        environment * logk_config.get('band_norm', 140), bandwidth * logk_config.get('band_norm', 140), b_reward * 2 + a_reward))
    print("b_reward {}, a_reward {}".format(b_reward, a_reward))
    return b_reward + a_reward


if __name__ == "__main__":
    config = {
        **logk_config
    }
    model = AdaptModel([0, 30]) \
        .add('head', [1, 100], 'discrete', 1) \
        .add('threshold', [100, 1000], 'discrete', 50) \
        .add_degrade(degrade_func) \
        .add_evaluate(evaluate_func) \
        .add_profile('../models/logk') \
        .add_config(config) \
        .build()

    model.train(0)
