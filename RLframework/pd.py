from adaptmodel import AdaptModel, AWStream
from image_interface import visiontask2 as visiontask
import numpy as np
import time
from utils.config import pd_config


def pd_degrade(res, skip, quant):
    return visiontask(res, (1, 1051), int(skip), 'train', int(quant), False)


def pd_degrade_test(res, skip, quant):
    return visiontask(res, (1, 1501), int(skip), 'test', int(quant), False)


def pd_evluate(env, bd, f1score):
    env *= pd_config['band_norm']
    bd *= pd_config['band_norm']
    f1score_loss = - abs(f1score - 1) * 1.5
    bandwidth_loss = -abs((bd-env) / max(bd, env))
    print("env {}, bd {}, reward {}".format(
        env, bd, f1score_loss + bandwidth_loss))
    return f1score_loss + bandwidth_loss


def test_awstream():
    config = {
        'degrade_test': pd_degrade_test,
        'band_norm': pd_config['band_norm'],
        'degrade': pd_degrade,
        'profile_path': '/home/shen/research/RL/WANStream/pd_profile',
        'cpu_num': pd_config['cpu_num']
    }
    model = AWStream() \
        .add('res', np.array([320, 640, 960, 1280, 1600, 1920])/1920) \
        .add('skip', np.array([29, 14, 9, 5, 2, 0]) + 1) \
        .add('quant', (0, 10, 20, 30, 40, 50)) \
        .add_config(config)\
        .build()
    model.profiling()
    model.test()
    model.show()


if __name__ == "__main__":
    test_awstream()
    raise Exception
    config = {
        'degrade': pd_degrade,
        'evaluate': pd_evluate,
        'degrade_test': pd_degrade_test,
        'cpu_num': pd_config['cpu_num'],
        'band_norm': pd_config['band_norm']
    }
    a = AdaptModel([0, 40])\
        .add('res', [0.15, 1])\
        .add('skip', [1, 30], [1, 2, 3, 5, 6, 10, 15, 30])\
        .add('quant', [0, 50], 1)\
        .add_profile('/home/shen/research/RL/WANStream/pd_profile')\
        .add_config(config)\
        .build()
    a.test(15)
