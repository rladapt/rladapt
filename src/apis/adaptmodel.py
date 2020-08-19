import sys
sys.path.append("..")
from .agent import Agent
import numpy as np
from utils.log import get_logger
import logging
from random import gauss, shuffle
from multiprocessing import Pool, cpu_count, set_start_method, freeze_support
import multiprocessing
import matplotlib.pyplot as plt
import math
import os
import pickle
from utils.pickle_interface import savecsv
from utils import search
from collections import namedtuple
import random
from utils.config import pd_config


class Env():
    pass


class AdaptModel():
    def __init__(self, low, high):
        self.high = high
        self.low = low
        self.dimensions = {}
        self.debug = False
        self.is_train = False
        self.all_rewards = []
        self._last_reward = 0
        self._epoch = 0

    def add(self, _name, _range, _type):
        self.dimensions[_name] = [
            _name,
            _range,
            _type
        ]
        return self

    @ property
    def dnames(self):
        return list(self.dimensions.keys())

    def _checkd(self, name):
        '''
        For discrete dimensions, options are fixed,
        _range should declare the number of options

        For continuous dimensions,
        _range should declare the range in a form of
        [A, B]

        '''
        _, _range, _type = self.dimensions[name]
        if _type == 'discrete':
            assert isinstance(_range, int)
            print('{} Options: {}'.format(name, _range))
        else:
            assert len(_range) == 2
            print('{} Range: {}'.format(name, _range))

    def add_degrade(self, degrade_func):
        assert callable(degrade_func)
        self.df = degrade_func
        return self

    def add_evaluate(self, evaluate_func):
        assert callable(evaluate_func)
        self.ef = evaluate_func
        return self

    def add_config(self, config):
        for key, val in config.items():
            setattr(self, key, val)
        return self

    def build(self):
        print('Start Building ...')
        print('[1] Dimensions:')
        for name in self.dimensions:
            self._checkd(name)

        print('[2] Agent Initialization:')
        assert hasattr(self, 'df')
        assert hasattr(self, 'ef')

        env = Env()
        env.observation_space = np.zeros([1, ])
        env.action_space = np.zeros([len(self.dimensions), ])
        self.actor_lr = 0.002
        self.critic_lr = 0.002
        self.capacity = 500
        self.batch_size = 32
        self.cpu_num = 2
        self.iterations = 200
        self.band_norm = pd_config['band_norm']
        self.policy_freq = 2
        params = {
            'env': env,
            'gamma': 0,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'tau': 0.001,
            'capacity': self.capacity,
            'batch_size': self.batch_size,
            'cuda': True,
            'alpha': 0.2
        }

        self.agent = Agent(**params)

        print('Build Complete.')
        return self

    def train(self, start_from=1, end=500):
        print('Start Training')
        if start_from:
            self.agent.load(str(start_from), self.profile_path)
            self._epoch = start_from
        freeze_support()
        set_start_method('spawn', force=True)
        for epoch in range(start_from + 1, end):
            self.is_train = True
            self._reward = 0
            self._epoch = epoch
            p = Pool(self.cpu_num)
            bds = np.arange(self.low, self.high,
                            (self.high - self.low) / self.iterations) / self.band_norm
            bds = np.clip(np.random.normal(
                bds, 0.1 / self.band_norm), self.low, self.high)
            shuffle(bds)
            for index, bd in enumerate(bds):
                if self.debug:
                    self._train_step_callback(
                        self._train_step(self._epoch, index, bd, bd))
                else:
                    p.apply_async(
                        self._train_step, (self._epoch, index, bd, bd), callback=self._train_step_callback, error_callback=self.error_callback)
            p.close()
            p.join()
            print("Finish Epoch {}, Reward {}".format(
                epoch, self._reward))
            if epoch % 1 == 0:
                self.test()
        print('End Training')

    def error_callback(self, error):
        print(error)

    def test(self, start_from=None):
        self.test_results = [[] for _ in range(4)]
        self.test_cache = {}
        if start_from:
            self.agent.load(str(start_from), self.profile_path)
            self._epoch = start_from
        print("Start Testing")
        self.is_train = False
        freeze_support()
        set_start_method('spawn', force=True)
        p = Pool(self.cpu_num)
        interval = 1 if start_from else 5
        bds = np.arange(self.low, self.high,
                        (self.high - self.low) / self.iterations * interval) / self.band_norm
        shuffle(bds)
        for index, bd in enumerate(bds):
            p.apply_async(
                self._test_step, (self._epoch, index, bd, bd), callback=self._test_step_callback, error_callback=self.error_callback)
        p.close()
        p.join()
        self.plot()
        print("Finish Testing")

    def _test_step(self, epoch, index, bd, nxtbd):
        action, _ = self.predict(bd)
        if tuple(action) in self.test_cache:
            print("SKIPPED ACTION {}".format(action))
            _, _, result, reward = self.test_cache[tuple(action)]
            return action, bd, result, reward
        else:
            result = self.df(*action)
            reward = self.ef(bd, *result)
            print('[BD]: {} [ACTION] {}'.format(bd * self.band_norm, action))
            return action, bd, result, reward

    def _test_step_callback(self, result):
        action, env, (bd, acc), reward = result
        self.test_cache[tuple(action)] = result
        self.test_results[0].append(env)
        self.test_results[1].append(bd)
        self.test_results[2].append(acc)
        self.test_results[3].append(reward)
        return

    def _train_step(self, epoch, index, bd, nxtbd):
        action, ori_action = self.predict(bd)
        result = self.df(*action)
        reward = self.ef(bd, *result)
        return bd, ori_action, reward, nxtbd, index

    def _train_step_callback(self, result):
        bd, action, reward, nxtbd, index = result
        self.agent.put(bd, action, reward, bd)
        if index % self.policy_freq == 0:
            self.agent.learn()
        self._reward += reward
        print('=' * 10 + "Epoch {}, Iteration {}".format(self._epoch, index) + '=' * 10)
        # if index == self.iterations // 2 - 1:
        #     self.all_rewards.append(self._reward)
        #     self._last_reward = self._reward
        # if index == self.iterations - 1:
        #     self.all_rewards.append(self._reward - self._last_reward)

        if index and index % 50 == 0:
            self.agent.save(str(self._epoch), self.profile_path)
            print("saving current model....")

    def predict(self, bd):
        action = self.agent.act(bd)
        if self.is_train:
            action = np.clip(np.random.normal(
                action, 0.25 * 1 ** self._epoch), -0.5, 0.5)
            if random.random() < 0.05:
                action = np.random.random(len(self.dimensions)).clip(-0.5, 0.5)
        else:
            action = np.clip(action, -0.5, 0.5)
        ori_action = action
        action = self.choose_action(action)
        return action, ori_action

    def load(self, epoch=1):
        self.agent.load(str(epoch), self.profile_path)
        self._epoch = epoch
        return self

    def load_test_results(self, epoch=1):
        self._epoch = epoch
        with open(os.path.join(self.profile_path, 'test_reuslts{}.pickle'.format(self._epoch)), 'rb') as f:
            self.test_results = pickle.load(f)
        print("load test results{} success.".format(self._epoch))
        return self

    def choose_action(self, actions):
        _result = []
        # cast actions from [-0.5,0.5] to [0,1]

        actions = iter(list(actions + 0.5))

        for name in self.dimensions:
            _, _range, _type = self.dimensions[name]
            if _type == 'discrete':
                '''
                For discrete dismensions, _range means 0 to X possibilities.
                However, int() will floor the result, so take 0 to X + 1.
                '''
                _result.append(
                    min(_range,
                        int((1 + _range) * actions.__next__())
                        )
                )
            else:
                _result.append((_range[1] - _range[0])
                               * actions.__next__() + _range[0])
        return _result

    def _add_config(self, key, val):
        setattr(self, key, val)
        print("set {} to {}".format(key, val))

    def add_profile(self, profile_path):
        assert(isinstance(profile_path, str))
        if os.path.exists(profile_path):
            assert(os.path.isdir(profile_path))
        else:
            os.mkdir(profile_path)
        self._add_config('profile_path', profile_path)
        return self

    def plot(self):
        self.test_results = np.array(
            sorted(np.array(self.test_results).T, key=lambda i: i[1])).T
        self.serialize(self.test_results, os.path.join(
            self.profile_path, 'test_reuslts{}.pickle'.format(self._epoch)))
        plt.plot(self.test_results[0], label='env bandwidth')
        plt.plot(self.test_results[1], label='predicted bandwidth')
        plt.legend()
        plt.savefig(os.path.join(self.profile_path, 'bd_result') +
                    str(self._epoch), dpi=400)
        plt.close()
        plt.plot(self.test_results[2], label='acc')
        plt.ylim(ymin=0, ymax=1)
        plt.legend()
        plt.savefig(os.path.join(self.profile_path,
                                 'acc_result') + str(self._epoch), dpi=400)
        plt.close()
        # reward plot
        self.all_rewards.append(sum(self.test_results[3]))
        if hasattr(self, 'all_rewards'):
            plt.plot(self.all_rewards, label='reward per 200 iterations')
            plt.legend()
            plt.savefig(os.path.join(
                self.profile_path, 'all_rewards'), dpi=400)
            plt.close()

    def serialize(self, obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    '''
    implementation of AWStream method
    
    parameters should be a subscrible array
    parameters[0] is for the first parameter
    ...


    '''

    def AWStreamTest(self, parameters, filename, actions=None):
        self._epoch = filename

        combinations = []

        def dfs(index, cur):
            if index == len(parameters):
                combinations.append(cur)
                return
            for param in parameters[index]:
                _cur = cur + [param]
                dfs(index + 1, _cur)
        if actions:
            combinations = actions
        else:
            dfs(0, [])
        # initialize test_results
        self.test_results = [[] for i in range(5)]
        freeze_support()
        set_start_method('spawn', force=True)
        p = Pool(self.cpu_num)
        for index, action in enumerate(combinations):
            p.apply_async(
                self._AWStreamTestStep, ([action]), callback=self._AWStreamTestCallback, error_callback=self.error_callback)
        p.close()
        p.join()
        self.plot()
        print("Finish Testing AWStream")
        self._epoch = 0

    def _AWStreamTestStep(self, action):
        result = self.df(*action)
        bd, acc = result
        print('[BD]: {} [ACTION]: {} [ACC]: {}'.format(
            bd * self.band_norm, action, result[-1]))
        return action, result

    def _AWStreamTestCallback(self, result):
        action, (bd, acc) = result
        self.test_results[0].append(bd)
        self.test_results[1].append(bd)
        self.test_results[2].append(acc)
        self.test_results[3].append(0)
        self.test_results[4].append(action)
        return
