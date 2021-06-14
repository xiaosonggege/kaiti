#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: meta_learning_pretrain
@time: 2021/5/5 4:40 下午
'''
import tensorflow as tf
import numpy as np
from neural_work.signal_data import Signal
import os

class Meta_process:
    def __init__(self, data:np.ndarray=None, epoch:int=None, batch_size:int=None,
                 support_Dataset:tf.data.Dataset=None,
                 query_Dataset:tf.data.Dataset=None):
        self._data_raw = data
        self._weights = dict()
        self._batch_size = batch_size
        self._epoch = epoch
        # self._Dataset = tf.data.Dataset.from_tensor_slices(tensors=(self._data_raw, self._data_raw)) \
        #     .batch(batch_size=self._batch_size) \
        #     .repeat(self._epoch)
        self._support_Dataset = support_Dataset
        self._support_iterator = self._support_Dataset.make_initializable_iterator()
        self._support_next_batch = self._support_iterator.get_next()

        self._query_Dataset = query_Dataset
        self._query_iterator = self._query_Dataset.make_initializable_iterator()
        self._query_next_batch = self._query_iterator.get_next()
        meta_lr = 1e-2 #gai
        self._optimizer = tf.train.AdamOptimizer(learning_rate=meta_lr)  # meta学习力度要大

    @property
    def support_Dataset(self):
        return self._support_Dataset

    @support_Dataset.setter
    def support_Dataset(self, dataset: tf.data.Dataset = None):
        self._Dataset = dataset

    @property
    def query_Dataset(self):
        return self._query_Dataset

    @query_Dataset.setter
    def query_Dataset(self, dataset:tf.data.Dataset=None):
        self._query_Dataset = dataset

    def support_init(self, sess:tf.Session):
        self._support_iterator.initializer.run(session=sess)

    def query_init(self, sess:tf.Session):
        self._query_iterator.initializer.run(session=sess)

    def config_weights(self, input_size:int, *layers):
        with tf.variable_scope(name_or_scope='weights', reuse=tf.AUTO_REUSE):
            for i, out_size in enumerate(layers):
                self._weights['w%s' % i] = tf.get_variable(
                    shape=(input_size, out_size),
                    name='w%s' % i,
                    dtype=tf.float64
                )
                self._weights['b%s' % i] = tf.get_variable(
                    shape=(out_size),
                    name='b%s' % i,
                    dtype=tf.float64
                )
                input_size = out_size

    @property
    def weights(self):
        return self._weights
    @weights.setter
    def weights(self, weights:dict):
        self._weights = weights

    def _dnn_forward(self, data_batch: tf.Tensor, weights:dict):
        '''

        :param data_batch:
        :return:
        '''
        output = None
        for i in range(len(weights) // 2):
            if output is None:
                output = tf.matmul(data_batch, weights['w%s' % i]) + weights['b%s' % i]
            else:
                output = tf.matmul(output, weights['w%s' % i]) + weights['b%s' % i]
            output = tf.nn.softplus(output)
        return output

    def meta_train(self, input_size:int, is_training:bool=True):
        '''
        meta learning
        :return:
        '''
        lr = 1e-3 #gai
        #网络参数结构需要和fune_training的网络结构相同
        layers = (100, 200, input_size)
        self.config_weights(input_size, *layers)
        support_x, support_y = self._support_next_batch
        query_x, query_y = self._query_next_batch
        output_supportset = self._dnn_forward(data_batch=support_x, weights=self._weights)
        self._loss = tf.reduce_mean(tf.square(output_supportset - support_y))

        grad = tf.gradients(ys=self._loss, xs=list(self._weights.values()))
        name2grad = dict(zip(self._weights.keys(), grad))
        self._fast_weights = dict(zip(self._weights.keys(),
                                  [self._weights[key] - lr * name2grad[key] for key in self._weights.keys()]))
        output_queryset = self._dnn_forward(data_batch=query_x, weights=self._fast_weights)
        self._query_loss = tf.reduce_mean(tf.square(output_queryset - query_y))
        opt_update = self._optimizer.minimize(loss=self._query_loss)

        saver = tf.train.Saver(self._weights)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if os.listdir(os.getcwd() + os.path.sep + 'checkpointfile'):
                saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(
                    os.getcwd() + os.path.sep + 'checkpointfile'))

            loss_optim = 1e9
            for epoch in range(self._epoch):
                self.support_init(sess=sess)
                self.query_init(sess=sess)
                query_total_loss = 0
                while True:
                    try:
                        sess.run(self._fast_weights)
                        _, query_loss = sess.run(fetches=[opt_update, self._query_loss])
                        # print(query_loss)
                        query_total_loss += query_loss
                    except tf.errors.OutOfRangeError:
                        break
                if epoch % 100 == 0:
                    print(query_total_loss)
                # if query_total_loss < loss_optim:
                #     loss_optim = query_total_loss
                #     saver.save(sess=sess, save_path=os.getcwd() + os.path.sep + \
                #         'checkpointfile' + os.path.sep + 'save_model', write_meta_graph=True)
                dict_res = self._optimizer.compute_gradients(self._loss, var_list=self._weights)
                dict_res_ = sess.run(dict_res)
        print('dict:是', dict_res_)
        return dict_res_

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer:tf.train.Optimizer):
        self._optimizer = optimizer


if __name__ == '__main__':
    # 内建去噪数据
    sg = Signal(dataset_size=100, feature_size=20)
    dataset = np.hstack((sg.dataset, sg.dataset))
    # # print(dataset.shape)
    from sindata_example import SinusoidGenerator, generate_dataset
    s = SinusoidGenerator(K=20)
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
