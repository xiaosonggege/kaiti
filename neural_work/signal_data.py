#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: main2
@time: 2020/10/8 3:02 下午
'''

import numpy as np

class Signal:
    rng = np.random.RandomState(0)
    def __init__(self, dataset_size:int, feature_size:int):
        '''
        :param dataset_size: 样本量
        :param feature_size: 单个样本的长度
        '''
        self._dataset_size = dataset_size
        self._feature_size = feature_size
        self.signal_maker()
        self.dataset_maker()

    @property
    def dataset_size(self):
        return self._dataset_size

    @dataset_size.setter
    def dataset_size(self, size:int):
        self._dataset_size = size

    @property
    def feature_size(self):
        return self._feature_size

    @feature_size.setter
    def feature_size(self, size):
        self._feature_size = size

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, signal:np.ndarray):
        self._signal = signal

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset:np.ndarray):
        self._dataset = dataset

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x:np.ndarray):
        self._x = x

    def signal_maker(self):
        '''
        自制数据
        :return:
        '''
        self._x = np.linspace(-10, 10, self._dataset_size * self._feature_size)
        # rng = np.random.RandomState(1)
        # self._x = rng.uniform(-10, 10, size=self._dataset_size * self._feature_size)
        self._signal = 3 * np.sin(self._x) + Signal.rng.random(size=self._x.shape)

    def dataset_maker(self):
        '''
        将一份数据拆分成shape=(self._dataset_size, self._feature_size)的二维数据集
        :return:
        '''
        self._dataset = self._signal.reshape(self._dataset_size, self._feature_size)

if __name__ == '__main__':
    sg = Signal(100, 20)
    print(sg.signal.shape)
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    # ax.plot(sg.x, sg.signal)
    ax.plot(sg.x, sg.signal)
    fig.show()
