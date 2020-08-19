'''

Generate `bandwidth--accuracy` data

For all awstream models

First, generate primary profiles from given parameters

Second, select profiles on pareto boundary

Third, test on test dataset

Forth, output formed data

'''

PROFILE_PATH = '../experiemnt/awstream'
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


from logk import degrade_func as logk_degrade
from logk import evaluate_func as logk_evaluate
from logk import degrade_func_test as logk_degrade_test
from logk import logk_config

PARAMETERS = {
    'pd': [
        np.array([320, 640, 960, 1280, 1600, 1920]) / 1920,
        [0, 2, 4, 5, 6, 7],
        [0, 10, 20, 30, 40, 50]
    ],
    'ar': [
        np.array([320, 640, 960, 1280, 1600, 1920]) / 1920,
        [0, 2, 4, 5, 6, 7],
        [0, 10, 20, 30, 40, 50]
    ],

}

JetStream_PARAMETERS = {
    'pd': [
        [0.5, 0.75],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [10]
    ],
    'ar': [
        [0.5, 0.75],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [10]
    ],
    'logk': [
        [50, 75],
        [100, 300, 500, 700, 900]
    ]
}


def primary_profiles(application, filename, test=False, run_test=False, actions=None, jetstream=False):
    if application not in ['pd', 'ar', 'logk']:
        raise Exception('Wrong Application Name')

    if actions or test:
        degrade = eval(application + '_degrade_test')
    else:
        degrade = eval(application + '_degrade')
    evaluate = eval(application + '_evaluate')

    model = AdaptModel(0, 20)

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

    if application == 'logk':
        model = model \
            .add('head', 10, 'discrete') \
            .add('threshold', 10, 'discrete') \
            .add_profile('../models/logk') \

    model = model \
        .add_degrade(degrade) \
        .add_evaluate(evaluate)\
        .build()

    if run_test and not jetstream:
        model.AWStreamTest(PARAMETERS[application], filename,
                           actions=actions)
    if run_test and jetstream:
        model.AWStreamTest(JetStream_PARAMETERS[application], filename,
                           actions=actions)
    model.load_test_results(filename)
    model.plot()

    '''
    test results for 200 iterations

    test_results[0] is environment bandwdith (normalized to 1)
    test_results[1] is bandwidth using predicted parameters (normalized to 1)
    test_results[2] is accuracy using predicted parameters (normalized to 1)
    test_results[3] is reward using predicted parameters (without normalization)

    '''
    return model.test_results


def transfer(test_results):
    test_results = test_results.T
    res = np.zeros((len(test_results), 4))
    index = 0
    actions = []
    for item in test_results:
        item[0] = 0
        item[1] *= eval(APPLICATION + '_config')['band_norm']
        res[index] = item[:-1]
        index += 1
        actions.append(item[-1])
    return res, actions


def pareto_profiles(test_results):
    deleted = []
    test_results = test_results.T
    for item in reversed(test_results):
        env, bd, acc, _, _ = item
        for item2 in reversed(test_results):
            _, bd2, acc2, _, _ = item2
            if acc2 > acc and bd2 <= bd:
                deleted.append((bd, acc))
                break
    new_test_results = np.zeros(
        (len(test_results) - len(deleted), 4))
    discarded_results = np.zeros(
        (len(deleted), 4)
    )
    actions = []
    actions_discard = []
    index = 0
    index_discard = 0
    for item in test_results:
        if (item[1], item[2]) not in deleted:
            item[0] = 0
            item[1] *= eval(APPLICATION + '_config')['band_norm']
            new_test_results[index] = item[:-1]
            actions.append(item[-1])
            index += 1
        else:
            item[0] = 0
            item[1] *= eval(APPLICATION + '_config')['band_norm']
            discarded_results[index_discard] = item[:-1]
            index_discard += 1
            actions_discard.append(item[-1])
    return new_test_results, discarded_results, actions, actions_discard


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
    # APPLICATION = 'ar'
    # np.set_printoptions(suppress=True)
    # stage1 = primary_profiles(APPLICATION, 'AWStream',
    #                           test=False, run_test=False)
    # stage2, stage2_discard, actions, actions_discard = pareto_profiles(stage1)
    # stage3 = primary_profiles(
    #     APPLICATION, 'AWStreamTest', test=True, run_test=True, actions=actions)
    # discarded = primary_profiles(
    #     APPLICATION, 'AWStreamTestDiscard', test=True, run_test=True, actions=actions_discard)
    # discarded, _ = transfer(discarded)
    # print(stage3)
    # stage4, _, _, _ = pareto_profiles(stage3)
    # stage3, _ = transfer(stage3)
    # formed_profiles(stage3, discarded)

    APPLICATION = 'ar'
    PROFILE_PATH = '/home/shen/research/RL/rladapt/experiemnt/jetstream'
    np.set_printoptions(suppress=True)
    stage1 = primary_profiles(APPLICATION, 'JetStream',
                              True, True, jetstream=True)
    stage1, _, _, _ = pareto_profiles(stage1)
    formed_profiles(stage1, stage1)
