'''

Generate `bandwidth--accuracy` data

For all rladapt models

First, generate primary profiles in [0, 40] Mbps

Second, select profiles on pareto boundary

Thrid, output formed data

'''

PROFILE_PATH = '../experiemnt/rladapt'
MODELS = {'pd': 40, 'ar': 7}
BD = {'pd': 40, 'ar': 40}
APPLICATION = ''

import sys
sys.path.append('.')
import numpy as np
import os

from apis.adaptmodel import AdaptModel
from pd import degrade_func as pd_degrade
from pd import evaluate_func as pd_evaluate
from pd import degrade_func_test as pd_degrade_test
from pd import pd_config

from ar import degrade_func as ar_degrade
from ar import evaluate_func as ar_evaluate
from ar import degrade_func_test as ar_degrade_test
from ar import ar_config


def primary_profiles(application, run_test=False, test=False):
    if application not in ['pd', 'ar', ' topk']:
        raise Exception('Wrong Application Name')

    if test:
        degrade = eval(application + '_degrade_test')
    else:
        degrade = eval(application + '_degrade')
    evaluate = eval(application + '_evaluate')

    model = AdaptModel(0, BD[application])

    if application == 'pd':
        model = model \
            .add('res', (0.15, 1), 'continuous')\
            .add('skip', 7, 'discrete')\
            .add('quant', 50, 'discrete')\
            .add_profile('../models/pd')
    if application == 'ar':
        model = model \
            .add('res', (0.15, 1), 'continuous')\
            .add('skip', 7, 'discrete')\
            .add('quant', 50, 'discrete')\
            .add_profile('../models/ar') \

    model = model \
        .add_degrade(degrade) \
        .add_evaluate(evaluate)\
        .build()

    if run_test:
        model.test(MODELS[application])
    model.load_test_results(MODELS[application])
    model.plot()

    '''
    test results for 200 iterations

    test_results[0] is environment bandwdith (normalized to 1)
    test_results[1] is bandwidth using predicted parameters (normalized to 1)
    test_results[2] is accuracy using predicted parameters (normalized to 1)
    test_results[3] is reward using predicted parameters (without normalization)

    '''
    return model.test_results


def pareto_profiles(test_results):
    deleted = []
    test_results = test_results.T
    for item in reversed(test_results):
        env, bd, acc, _ = item
        for item2 in reversed(test_results):
            _, bd2, acc2, _ = item2
            if acc2 > acc and bd2 <= bd:
                deleted.append(env)
                break
    new_test_results = np.zeros((len(test_results) - len(deleted), 4))
    discarded_results = np.zeros(
        (len(deleted), 4)
    )
    index = 0
    index_discard = 0
    for item in test_results:
        item[1] *= eval(APPLICATION + '_config')['band_norm']
        if item[0] not in deleted:
            item[0] = item[-1] = 0
            new_test_results[index] = item
            index += 1
        else:
            item[0] = item[-1] = 0
            discarded_results[index_discard] = item
            index_discard += 1
    return new_test_results, discarded_results


def formed_profiles(fig1, fig1_discard):
    fig1 = np.unique(fig1, axis=0)
    fig1_discard = np.unique(fig1_discard, axis=0)
    with open(os.path.join(PROFILE_PATH, 'fig1-{}.csv'.format(APPLICATION)), 'w') as f:
        f.write('x y\n')
        for item in fig1:
            _, bd, acc, _ = item
            if bd > 30:
                continue
            f.write('{} {}\n'.format(bd, acc))
    with open(os.path.join(PROFILE_PATH, 'fig1_discard-{}.csv'.format(APPLICATION)), 'w') as f:
        f.write('x y\n')
        for item in fig1_discard:
            _, bd, acc, _ = item
            if bd > 30:
                continue
            f.write('{} {}\n'.format(bd, acc))


if __name__ == "__main__":
    APPLICATION = 'pd'
    np.set_printoptions(suppress=True)
    stage1 = primary_profiles(APPLICATION, True, True)
    print(stage1)
    stage2, discarded = pareto_profiles(stage1)
    formed_profiles(stage2, discarded)
