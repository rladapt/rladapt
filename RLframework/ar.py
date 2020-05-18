from utils.config import ar_config
from ar_interface import artask, net
from utils.darknet import load_net
from utils.log import get_logger
from adaptmodel import AdaptModel, AWStream
import numpy as np


logger = get_logger(__name__)
BATCH_SIZE = 30


def degrade(resolution, skip, quantizer):
    bd, f1score = artask(net.net, resolution, BATCH_SIZE,
                         skip, 'train', quantizer)
    return bd, f1score


def degrade_test(resolution, skip, quantizer):
    return artask(net.net, resolution, (1, 101),
                  skip, 'test', quantizer)


def evaluate(env, bd, f1score):
    env *= ar_config['band_norm']
    bd *= ar_config['band_norm']
    f1score_loss = - abs(f1score - 1) * 1.5
    bandwidth_loss = -abs((bd-env) / max(bd, env)) * 1
    loss = f1score_loss + bandwidth_loss
    logger.info('[ENV] {}, [BD] {}, [F1score] {}, [REWARD] {}'.format(
        env, bd, f1score, loss))
    return loss


def test_awstream():
    config = {
        'degrade_test': degrade_test,
        'band_norm': 161,
        'degrade': degrade,
        'profile_path': '/home/shen/research/RL/WANStream/ar_profile'
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
    config = {
        **ar_config,
        'degrade': degrade,
        'evaluate': evaluate,
        'degrade_test': degrade_test,
    }
    model = AdaptModel((0, 20)) \
        .add('resolution', (0.1, 1)) \
        .add('skip', [1, 30], [1, 2, 3, 5, 6, 10, 15, 30])\
        .add('quant', [0, 50], 1)\
        .add_config(config) \
        .add_profile('/home/shen/research/RL/WANStream/ar_profile_2') \
        .build()

    model.train(0)
    # model.test(8)
