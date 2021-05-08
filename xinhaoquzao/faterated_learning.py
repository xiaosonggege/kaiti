#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: faterated_learning
@time: 2021/5/6 7:44 下午
'''
import tensorflow as tf
import numpy as np
import multiprocessing
from sindata_example import SinusoidGenerator, generate_dataset
from signal_data import Signal
from processQuzao import ProcessQuzao
from meta_learning_pretrain import Meta_process

# TODO 利用A3C算法思路构建多线程联邦学习以及应用黄科力的论文思路复现
class Fl:
    def __init__(self, datas:list):
        self._datas = datas
        self._gradient_dict = multiprocessing.Manager().dict()
        # self._Meta_learn = Meta_process(epoch=epoch, support_Dataset=)

    def faterated_sub_training(self, mp:Meta_process, input_size:int, gradient_dict:dict):
        gradient_sub_dict = mp.meta_train(input_size=input_size)
        for grad_key in gradient_dict.keys():
            gradient_dict[grad_key] += gradient_sub_dict[grad_key]


    def faterated_training(self, epoch:int):
        jobs = []
        self._optimizer = None
        for support_Dataset, query_Dataset in self._datas:
            pq = Meta_process(epoch=epoch, support_Dataset=support_Dataset, query_Dataset=query_Dataset)
            if self._optimizer is None:
                self._optimizer = pq.optimizer
            jobs.append(multiprocessing.Process(target=pq.meta_train, args=(pq, 20, self._gradient_dict)))
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        for key in self._gradient_dict.keys():
            self._gradient_dict[key] /= len(self._datas)
        self._optimizer.apply_gradients(grads_and_vars=self._gradient_dict.items())


if __name__ == '__main__':
    train_ds, test_ds = generate_dataset(K=20)
    x_train, y_train = np.split(train_ds, 2, axis=-1)
    x_train = x_train.reshape(20000, -1)
    y_train = y_train.reshape(20000, -1)

    x_test, y_test = np.split(test_ds, 2, axis=-1)
    x_test = x_test.reshape(1000, -1)
    y_test = y_test.reshape(1000, -1)
    train_ds_Dataset = tf.data.Dataset.from_tensor_slices(tensors=(x_train, y_train)).batch(512)
    test_ds_Dataset = tf.data.Dataset.from_tensor_slices(tensors=(x_test, y_test)).batch(25)
    pq = Meta_process(epoch=500000, support_Dataset=train_ds_Dataset, query_Dataset=test_ds_Dataset)
    # pq.Dataset = train_ds_Dataset
    pq.meta_train(input_size=20)
