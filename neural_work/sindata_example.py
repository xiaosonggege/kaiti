#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: sindata_example
@time: 2021/5/4 4:09 下午
'''
import numpy as np
from matplotlib import pyplot as plt

class SinusoidGenerator():
    '''
        Sinusoid Generator.

        p(T) is continuous, where the amplitude varies within [0.1, 5.0]
        and the phase varies within [0, π].

        This abstraction is the basically the same defined at:
        https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0
    '''

    def __init__(self, K=10, amplitude=None, phase=None):
        '''
        Args:
            K: batch size. Number of values sampled at every batch.
            amplitude: Sine wave amplitude. If None is uniformly sampled from
                the [0.1, 5.0] interval.
            pahse: Sine wave phase. If None is uniformly sampled from the [0, π]
                interval.
        '''
        self.K = K
        self.amplitude = amplitude if amplitude else np.random.uniform(0.1, 5.0)
        self.phase = phase if amplitude else np.random.uniform(0, np.pi)
        self.sampled_points = None
        self.x = self._sample_x()

    def _sample_x(self):
        return np.random.uniform(-5, 5, self.K)  # shape=(self.K, )

    def f(self, x):
        '''Sinewave function.'''
        return self.amplitude * np.sin(x - self.phase)  # shape=x.shape

    def batch(self, x=None, force_new=False):
        '''Returns a batch of size K.

        It also changes the sape of `x` to add a batch dimension to it.

        Args:
            x: Batch data, if given `y` is generated based on this data.
                Usually it is None. If None `self.x` is used.
            force_new: Instead of using `x` argument the batch data is
                uniformly sampled.

        '''
        if x is None:
            if force_new:
                x = self._sample_x()
            else:
                x = self.x
        y = self.f(x)
        return x[:, None], y[:, None]  # x_shape=(x.shape[0], 1), y_shape=(y.shape[0], 1)
        # rng = np.random.RandomState(0)
        # noise = rng.random(size=y[:, None].shape)
        # return y[:, None]+noise, y[:, None]+noise
    def equally_spaced_samples(self, K=None, force_new=False):
        '''Returns `K` equally spaced samples.'''
        if K is None:
            K = self.K
        return self.batch(x=np.linspace(-5, 5, K),
                          force_new=force_new)  # x_shape=(x.shape[0], 1), y_shape=(y.shape[0], 1)
        # 输出的是一维信号，由横轴采样点计算出y值，而后返回x, y的维度是增加了一维


def plot(data, *args, **kwargs):
    '''Plot helper.'''
    x, y = data
    return plt.plot(x, y, *args, **kwargs)


def generate_dataset(K, train_size:int=20000, test_size:int=1000, return_ndarray:bool=True):
    '''Generate train and test dataset.

    A dataset is composed of SinusoidGenerators that are able to provide
    a batch (`K`) elements at a time.
    '''

    def _generate_dataset(size):
        return [SinusoidGenerator(K=K) for _ in range(size)]  # 维度是[shape(x.shape[0], 1)]

    def _genersate_dataset_ndarray(size:int):
        dataset = None
        for i in range(size):
            dataset = np.hstack(SinusoidGenerator(K=K).batch())[None, :] if dataset is None else \
                np.concatenate((dataset, np.hstack(SinusoidGenerator(K=K).batch())[None, :]), axis=0)
        return dataset

    return (_generate_dataset(train_size), _generate_dataset(test_size)) if not return_ndarray else \
        (_genersate_dataset_ndarray(train_size), _genersate_dataset_ndarray(test_size))

if __name__ == '__main__':
    # for _ in range(3):
    #     plt.title('Sinusoid examples')
    #     plot(SinusoidGenerator(K=100).equally_spaced_samples())
    # plt.show()
    # s = SinusoidGenerator(K=5)
    # x, y = s.equally_spaced_samples(force_new=True)
    # print(x.shape, y.shape)
    # plt.plot(x, y)
    # plt.show()
    train_ds, test_ds = generate_dataset(K=10)
    print(train_ds.shape, test_ds.shape) #(20000, 10, 2) (10, 10, 2)
