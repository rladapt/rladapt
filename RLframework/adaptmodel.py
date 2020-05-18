from agent import Agent
# from sac.sac import SAC
import numpy as np
from utils.log import get_logger
import logging
from random import gauss, shuffle
from degrade import *
from multiprocessing import Pool, cpu_count, set_start_method, freeze_support
import multiprocessing
from utils import search
from utils.pickle_interface import savecsv
import matplotlib.pyplot as plt
import math
import os
import pickle
import math


class Env():
    pass


class AdaptModel():
    def __init__(self, bandvar):
        assert(len(bandvar) == 2 and isinstance(bandvar[0], (int, float)) and
               isinstance(bandvar[1], (int, float)) and bandvar[0] < bandvar[1])
        self.knobs = []
        self.range = []
        self.discret = []
        self.bandvar = bandvar
        self.logger = get_logger('AdaptModel', level=logging.INFO)
        self.TRAIN = False
        self._epoch = 0

        # constants
        self.CONTINUOUS = 1
        self.DISCRET = 2

        # modifiable config
        default_config = {
            'epochs': 500,
            'change': 0.1,
            'band_norm': 140,
            'lr_decay': math.sqrt(0.62),
            'test_interval': 2,
            'cpu_num': 4,
            'actor_lr': 0.001,
            'critic_lr': 0.001,
            'batch_size': 32,
            'rand_vars_3': [0.1],
            'capacity': 1000,
            'cpu_num_test': 4
        }
        self.add_config(default_config)

        # logging
        self.logger.info("bandwidth varies from {} to {}".format(
            bandvar[0], bandvar[1]))

        # mem
        self.mem = {}
        self.mem_i = 0

    def add(self, knob, _range: list, discret=None):
        assert(_range.__len__() == 2)
        assert(isinstance(_range[0], (int, float)))
        assert(isinstance(_range[1], (int, float)))
        assert(_range[0] <= _range[1])
        assert(isinstance(knob, str))
        if discret is not None:
            assert(isinstance(discret, (int, float, list, tuple)))
        self.knobs.append(knob)
        self.range.append(_range)
        self.discret.append(discret)

        # logging
        self.logger.info(
            "add {}, range {}, discret {}".format(knob, _range, discret))

        return self

    def add_degrade(self, degrade):
        assert(callable(degrade))
        self.degrade = degrade
        return self

    def add_evaluate(self, evaluate):
        assert(callable(evaluate))
        self.evaluate = evaluate
        return self

    def add_profile(self, profile_path):
        assert(isinstance(profile_path, str))
        if os.path.exists(profile_path):
            assert(os.path.isdir(profile_path))
        else:
            os.mkdir(profile_path)
        self.profile_path = profile_path
        return self

    def add_params(self, params):
        self.params = params
        return self

    def build(self):
        assert(self.evaluate and self.degrade)
        env = Env()
        env.observation_space = np.zeros([1, ])
        env.action_space = np.zeros([len(self.knobs), ])
        _params = {}
        if hasattr(self, 'params'):
            _params = self.params
        params = {
            **_params,
            'env': env,
            'gamma': 0,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'tau': 0.02,
            'capacity': self.capacity,
            'batch_size': self.batch_size
        }
        if hasattr(self, 'use_sac'):
            self.agent = SAC(**params)
            print('using SAC as backend')
        else:
            self.agent = Agent(**params)

        # serialize action range
        for index, knob in enumerate(self.knobs):
            if self.discret[index] is not None and isinstance(self.discret[index], (int, float)):
                self.range[index] = self._serialize(self.range[index],
                                                    (self.range[index][1] - self.range[index][0]) // self.discret[index] + 1)
            elif self.discret[index] is not None  \
                    and isinstance(self.discret[index], (list, tuple)):
                self.range[index] = list(self.discret[index])
            else:
                # contingence
                self.range[index] = tuple(self.range[index])
        for knob in self.knobs:
            setattr(self, knob, 0.5)

        # logging
        self.logger.info("build complete")

        return self

    def add_config(self, config):
        assert(isinstance(config, dict))
        for key, value in config.items():
            self._add_config(key, value)
        return self

    def _add_config(self, key, val):
        setattr(self, key, val)
        self.logger.info("set {} to {}".format(key, val))

    def _serialize(self, x, length):
        return list(np.linspace(min(x), max(x), length))

    def _choose_action(self, action):
        _result = []
        for index, a in enumerate(action):
            if isinstance(self.range[index], tuple):
                _result.append((self.range[index][1] - self.range[index][0]) * (
                    a + getattr(self, self.knobs[index])) + self.range[index][0])
            else:
                _result.append(
                    self.range[index][int(-0.5 + len(self.range[index]) * (a + getattr(self, self.knobs[index])))])
        return _result

    def predict(self, bd):
        bd = bd / self.band_norm
        action = self.agent.act(bd)
        for i in range(len(action)):
            action[i] = max(min(0.5, action[i]), -0.5)
        action = self._choose_action(action)
        return action

    def load(self, num=1):
        self.agent.load(str(num), self.profile_path)
        return self

    def test(self, start_from=None):
        if self.TRAIN == False:
            try:
                assert(start_from)
                self._epoch = start_from
                self.agent.load(str(start_from), self.profile_path)
            except Exception as e:
                self.logger.info(e)
                return
        # logging
        self.logger.info("start testing...")
        self.result = [[], []]
        self.f1score = [[], []]
        freeze_support()
        set_start_method('spawn', force=True)
        p = Pool(self.cpu_num_test)
        bds = np.arange(self.bandvar[0], self.bandvar[1], 0.2)
        actions = []
        _bds = []
        _valid = []
        for bd in bds:
            bd = bd / self.band_norm
            action = self.agent.act(bd)
            action = [max(min(0.5, i), -0.5) for i in action]
            action = self._choose_action(action)
            action = tuple(action)
            if action not in actions:
                _valid.append(1)
            else:
                _valid.append(0)
            actions.append(action)
            _bds.append(bd)
        #  bd, f1score, env, _
        for index, action in enumerate(actions):
            if _valid[index] == 0:
                self._test_callback((-1, -1, _bds[index], -1))
                continue
            p.apply_async(
                self._test_step, (index, _bds[index], action), callback=self._test_callback)
        p.close()
        p.join()
        # self.tuple_sort()
        self.show()
        self.logger.info("end testing...")

    def train(self, start_from=1, end=None):
        if end == None:
            end = self.epochs
        self.TRAIN = True
        try:
            self.agent.load(str(start_from), self.profile_path)
            self.agent.actor_lr *= self.lr_decay ** max(1, (start_from-1))
            self.agent.critic_lr *= self.lr_decay ** max(1, (start_from-1))
        except Exception as e:
            self.logger.info(e)
            self.logger.info("no model, start new")
        # logging
        self.logger.info(
            "start training {} epochs".format(end - start_from - 1))
        self.RAND_VARS = [0.3, 0.3, 0.3]
        freeze_support()
        set_start_method('spawn', force=True)
        for epoch in range(start_from + 1, end):
            self._reward = 0
            self._epoch = epoch
            self.agent.critic_lr = max(1e-4, self.agent.critic_lr)
            self.agent.actor_lr = max(1e-4,  self.agent.actor_lr)
            if epoch > 3:
                self.RAND_VARS = self.rand_vars_3
            p = Pool(self.cpu_num)
            bds = list(gauss(i, self.change / 2)
                       for i in np.arange(self.bandvar[0], self.bandvar[1], 0.1))
            shuffle(bds)
            for index, bd in enumerate(bds):
                nxtbd = bds[(index+1) % len(bds)]
                p.apply_async(
                    self._single_step, (epoch, index, bd, nxtbd), callback=self._single_step_callback)

                # self._single_step(epoch, index, bd, nxtbd)
            p.close()
            p.join()
            self.logger.info("finish epoch {}, reward {}".format(
                epoch, self._reward))
            if (epoch and epoch % self.test_interval == 0):
                self.test()
            self.agent.actor_lr *= self.lr_decay
            self.agent.critic_lr *= self.lr_decay

    def _single_step(self, epoch, index, bd, nxtbd):
        try:
            self.logger.debug('=' * 10, epoch, index, '=' * 10)
            bd = bd / self.band_norm
            action = self.agent.act(bd)
            self.logger.debug(action)
            action = np.clip(np.random.normal(
                action, self.RAND_VARS[int(len(self.RAND_VARS) * bd / self.bandvar[1] - 0.5)]), -0.5, 0.5)
            action = tuple(self._choose_action(action))
            if (bd, action) not in self.mem:
                result = self.degrade(*action)
                reward = self.evaluate(bd, *result)
            else:
                reward = self.mem[(bd, action)]
            self.logger.debug(bd * self.band_norm, action, reward, nxtbd)
        except Exception as e:
            print(e)
        return bd, action, reward, nxtbd, index

    def _test_step(self, index, bd, action):
        try:
            print('=' * 10, index, '=' * 10)
            if hasattr(self, 'degrade_test'):
                result = self.degrade_test(*action)
            else:
                result = self.degrade(*action)
        except Exception as e:
            print(e)
        return result[0], result[1], bd, action  # all normalized to [0,1]

    def _single_step_callback(self, result):
        try:
            bd, action, reward, nxtbd, index = result
            if (bd, tuple(action)) not in self.mem:
                self.mem[(bd, tuple(action))] = reward
            self.agent.put(bd, action, reward, bd)
            self.agent.learn()
            self._reward += reward
            if index and index % 30 == 0:
                self.agent.save(str(self._epoch), self.profile_path)
                self.logger.debug("saving current model....")
        except Exception as e:
            print(e)

    def _test_callback(self, result):
        try:
            bd, f1score, env, _ = result
            self.result[0].append(bd * self.band_norm)  # pred bd
            self.result[1].append(env * self.band_norm)  # gt bd
            self.f1score[0].append(f1score)  # pred acc
            self.f1score[1].append(0.9)  # gt acc
        except Exception as e:
            print(e)

    def load_pickle(self):
        with open(os.path.join(self.profile_path, 'result.pickle') + str(self._epoch), 'rb') as f:
            self.result = pickle.load(f)
        with open(os.path.join(self.profile_path, 'f1score.pickle') + str(self._epoch), 'rb') as f:
            self.f1score = pickle.load(f)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def tuple_sort(self):
        _tmp = []
        for i in range(len(self.result[0])):
            _tmp.append((self.result[0][i], self.result[1][i],
                         self.f1score[0][i], self.f1score[1][i]))
        _tmp = sorted(_tmp, key=lambda i: i[1])
        for index, i in enumerate(_tmp):
            # skipped, use last one(must exist and cannot be -1)
            if i[0] == -1 * self.band_norm:
                _i = i
                i = list(_tmp[index - 1])
                i[1] = _i[1]  # use his own env
                _tmp[index] = i
            self.result[0][index] = i[0]
            self.result[1][index] = i[1]
            self.f1score[0][index] = i[2]
            self.f1score[1][index] = i[3]

    def show(self):
        try:
            with open(os.path.join(self.profile_path,
                                   'result.pickle') + str(self._epoch), 'wb') as f:
                pickle.dump(self.result, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(self.profile_path,
                                   'f1score.pickle') + str(self._epoch), 'wb') as f:
                pickle.dump(self.f1score, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e)
        plt.plot(self.result[1], label='max bandwidth')
        plt.plot(self.result[0], label='predicted bandwidth')
        plt.legend()
        plt.savefig(os.path.join(self.profile_path, 'bd_result') +
                    str(self._epoch), dpi=400)
        plt.close()
        plt.plot(self.result[1], self.f1score[0], label='acc')
        plt.plot(self.result[1], self.f1score[1], label='0.9gt_')
        plt.ylim(ymin=0, ymax=1)
        plt.legend()
        plt.savefig(os.path.join(self.profile_path,
                                 'acc_result') + str(self._epoch), dpi=400)
        plt.close()

    def __repr__(self):
        return str(self.knobs) + '\n' + str(self.range)

    def __str__(self):
        return self.__repr__()


class AWStream():
    def __init__(self):
        self.logger = get_logger("AWStream")
        self.knobs = []
        self.names = []
        self.configs = []

    def add(self, name, knobs):
        assert isinstance(knobs, (tuple, list, np.ndarray))
        self.knobs.append(knobs)
        self.names.append(name)
        self.logger.info('added {}'.format(name))
        return self

    def build(self):
        self._build([], 0)
        self.logger.info('build complete')
        return self

    def _build(self, cur, level):
        if level == len(self.names):
            self.configs.append(cur)
            return
        for i in self.knobs[level]:
            self._build(cur + [i], level+1)

    def add_config(self, config):
        for key, val in config.items():
            setattr(self, key, val)
        return self

    def profiling(self):
        self.logger.info("start profiling...")
        self.bds = {}
        self.f1cores = {}
        self.configs = [tuple(i) for i in self.configs]
        freeze_support()
        set_start_method('spawn', force=True)
        p = Pool(self.cpu_num)
        for i in self.configs:
            p.apply_async(self._step, (i, 'train'),
                          callback=self._step_callback)
        p.close()
        p.join()
        to_remove = []
        for i in self.configs:
            # choose i, see if we need to eliminate this config
            for j in self.configs:
                if self.bds[i] > self.bds[j] and \
                        self.f1cores[i] < self.f1cores[j]:
                    to_remove.append(i)
                    break
        for i in to_remove:
            self.bds[i] = [self.bds[i], 'stale']
            self.f1cores[i] = [self.f1cores[i], 'stale']
        self.logger.info("end profiling...")

    def _step(self, config, mode):
        print(config)
        if mode == 'train':
            bd, f1score = self.degrade(*config)
        if mode == 'test':
            bd, f1score = self.degrade_test(*config)
        return bd, f1score, config, mode

    def _step_callback(self, result):
        bd, f1score, config, mode = result
        if mode == 'train':
            bd = bd * self.band_norm
            self.bds[config] = bd
            self.f1cores[config] = f1score
        if mode == 'test':
            bd = bd * self.band_norm
            if isinstance(self.bds[config], list):
                self.bds[config][0] = bd
                self.f1cores[config][0] = f1score
            else:
                self.bds[config] = bd
                self.f1cores[config] = f1score

    def test(self):
        self.logger.info('start testing...')
        freeze_support()
        set_start_method('spawn', force=True)
        p = Pool(self.cpu_num)
        for i in self.configs:
            p.apply_async(self._step, (i, 'test'),
                          callback=self._step_callback)
        p.close()
        p.join()
        self.logger.info('end testing...')

    def show(self):
        to_plot = []
        for i in self.bds:
            to_plot.append([self.bds[i], self.f1cores[i]])

        def _sort(i):
            if isinstance(i[0], list):
                return i[0][0]
            else:
                return i[0]
        tmp = sorted(to_plot, key=_sort)
        to_plot = [[], []]
        for i in tmp:
            to_plot[0].append(i[0])
            to_plot[1].append(i[1])
        with open(os.path.join(self.profile_path, 'awstream.pickle'), 'wb') as f:
            pickle.dump(to_plot, f, pickle.HIGHEST_PROTOCOL)
        # self.plot(to_plot)

    def plot(self, data):
        plt.plot(data[0], data[1], marker='o')
        plt.savefig(
            '/home/shen/research/RL/WANStream/ar_profile/awstream', dpi=400)


if __name__ == "__main__":

    prefix = 'logk_profile'
    assert os.path.isdir(prefix)

    def split(data):
        data1, data2 = [], []
        for i in data:
            if isinstance(i, list):
                data2.append(i[0])
            else:
                data1.append(i)
        return data1, data2

    def filter(data, threshold):
        _data = [[], []]
        for i in range(len(data[0])):
            if isinstance(data[0][i], list) and data[0][i][0] >= threshold[0] and data[0][i][0] <= threshold[1]:
                _data[0].append(data[0][i])
                _data[1].append(data[1][i])
            if isinstance(data[0][i], (int, float)) and data[0][i] >= threshold[0] and data[0][i] <= threshold[1]:
                _data[0].append(data[0][i])
                _data[1].append(data[1][i])
        return _data

    def unique(data):
        _data = []
        for i in data:
            if i not in _data:
                _data.append(i)
        return _data

    with open(os.path.join(prefix, 'awstream.pickle'), 'rb') as f:
        awstream = pickle.load(f)
    with open(os.path.join(prefix, 'result.pickle18'), 'rb') as f:
        rladapt_bd = pickle.load(f)
    with open(os.path.join(prefix, 'f1score.pickle18'), 'rb') as f:
        rladapt_f1 = pickle.load(f)
    awstream = filter(awstream, (0, 40))
    aw_bd_1, aw_bd_2 = split(awstream[0])
    aw_f1_1, aw_f1_2 = split(awstream[1])
    # tmp
    savecsv(aw_bd_1, aw_f1_1, os.path.join(prefix, 'awstream.opt.csv'))
    savecsv(aw_bd_2, aw_f1_2, os.path.join(prefix, 'awstream.stale.csv'))
    # tmp
    plt.plot(aw_bd_1, aw_f1_1,
             label='awstream', marker='^', c='#1e88e5', linewidth=3, alpha=0.7, markersize=8, linestyle='dotted')
    plt.scatter(aw_bd_2, aw_f1_2, 45,
                label='awstream drop', marker='^', c='grey')
    aw_func = np.poly1d(np.polyfit(aw_bd_1, aw_f1_1, 2))
    # plt.plot(aw_bd_1, aw_func(aw_bd_1), c='#1e88e5',
    #         linewidth=1.5, linestyle='dashed')
    rladapt = [[], []]
    for i in range(len(rladapt_bd[0])):
        if math.isnan(rladapt_f1[0][i]) == False and rladapt_bd[0][i] >= 0 and rladapt_bd[0][i] <= aw_bd_1[-1]:
            rladapt[0].append(rladapt_bd[0][i])
            rladapt[1].append(rladapt_f1[0][i])
    _rladapt = [[rladapt[0][i], rladapt[1][i]] for i in range(len(rladapt[0]))]
    _rladapt = unique(_rladapt)
    _rladapt = np.array(_rladapt)
    rladapt = _rladapt.T
    plt.plot(rladapt[0], rladapt[1], label='rl-adapt(ours)',
             marker='^', c='#f44336', linewidth=3, alpha=0.7, markersize=8, linestyle='dotted')
    # tmp
    savecsv(rladapt[0], rladapt[1], os.path.join(prefix, 'rl-adapt.csv'))
    # tmp
    rl_func = np.poly1d(np.polyfit(rladapt[0], rladapt[1], 2))
    # plt.plot(rladapt[0], rl_func(rladapt[0]),
    #         c='#f44336', linewidth=1.5, linestyle='dashed')
    plt.ylim(ymin=0.5, ymax=1)
    plt.xlim(xmin=0)
    plt.xlabel('bandwidth(mbps)')
    plt.ylabel('task accuracy')
    plt.title('PD task')
    plt.legend(loc='best')
    plt.savefig(os.path.join(prefix, 'compare.jpg'), dpi=600)
    # model.plot(result)
