#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: processQuzao
@time: 2021/3/10 8:13 下午
'''
import tensorflow as tf
import numpy as np
from neural_work.signal_data import Signal
import os

class ProcessQuzao:
    def __init__(self, data:np.ndarray=None, batch_size:int=100, epoch:int=None):
        self._data_raw = data
        self._encoder_weights = dict()
        self._decoder_weights = dict()
        self._batch_size = batch_size
        self._epoch = epoch
        self._Dataset = tf.data.Dataset.from_tensor_slices(tensors=(self._data_raw, self._data_raw))\
        .batch(batch_size=self._batch_size)\
        .repeat(self._epoch)
        self._iterator = self._Dataset.make_initializable_iterator()
        self._next_batch = self._iterator.get_next()

    @property
    def Dataset(self):
        return self._Dataset

    @Dataset.setter
    def Dataset(self, dataset:tf.data.Dataset=None):
        self._Dataset = dataset

    def init(self, sess:tf.Session):
        self._iterator.initializer.run(session=sess)

    def config_encoder_weights(self, input_size:int, *layers):
        with tf.variable_scope(name_or_scope='encoder', reuse=tf.AUTO_REUSE):
            for i, out_size in enumerate(layers):
                self._encoder_weights['w%s' % i] = tf.get_variable(
                    shape=(input_size, out_size),
                    name='w%s' % i,
                    dtype=tf.float64
                )
                self._encoder_weights['b%s' % i] = tf.get_variable(
                    shape=(out_size),
                    name='b%s' % i,
                    dtype=tf.float64
                )
                input_size = out_size

    def config_decoder_weights(self, input_size:int, *layers):
        '''
        解码器目前无用
        :param input_size:
        :param layers:
        :return:
        '''
        print(input_size)
        with tf.variable_scope(name_or_scope='decoder', reuse=tf.AUTO_REUSE):
            for i, out_size in enumerate(layers):
                self._decoder_weights['w%s' % i] = tf.get_variable(
                    shape=(input_size, out_size),
                    name='w%s' % i,
                    dtype=tf.float64
                )
                self._decoder_weights['b%s' % i] = tf.get_variable(
                    shape=(out_size),
                    name='b%s' % i,
                    dtype=tf.float64
                )
                input_size = out_size

    def encoder_forward(self, data_batch:tf.Tensor):
        '''
        :param data_batch:
        :return:
        '''
        encoder = None
        for i in range(len(self._encoder_weights) // 2):
            if encoder is None:
                encoder = tf.matmul(data_batch, self._encoder_weights['w%s' % i]) + self._encoder_weights['b%s' % i]
            else:
                encoder = tf.matmul(encoder, self._encoder_weights['w%s' % i]) + self._encoder_weights['b%s' % i]
            encoder = tf.nn.softplus(encoder)
        return encoder


    def decoder_forward(self, input:tf.Tensor):
        '''
        :param input:
        :return:
        '''
        decoder = None
        for i in range(len(self._decoder_weights) // 2):
            if decoder is None:
                decoder = tf.matmul(input, self._decoder_weights['w%s' % i]) + self._decoder_weights['b%s' % i]
            else:
                decoder = tf.matmul(decoder, self._decoder_weights['w%s' % i]) + self._decoder_weights['b%s' % i]
            decoder = tf.nn.softplus(decoder)
        return decoder


    def train_and_predict(self, input_size:int, is_training:bool=True):
        '''
        '''
        #编码解码器方式的初始化
        # encoder_layers = (100, 200)
        # # print(self._next_batch.get_shape().as_list())
        # self.config_encoder_weights(self._data_raw.shape[-1], *encoder_layers)
        # decoder_layers = (200, 100, self._data_raw.shape[-1])
        # self.config_decoder_weights(encoder_layers[-1], *decoder_layers)
        # x, y = self._next_batch
        # encoder_opt = self.encoder_forward(x)
        # decoder_opt = self.decoder_forward(encoder_opt)
        # self._loss = tf.reduce_mean(tf.square(decoder_opt - y))

        #无编解码器方式的初始化
        encoder_layers = (100, 200, input_size)
        self.config_encoder_weights(input_size, *encoder_layers)
        x, y = self._next_batch
        encoder_opt = self.encoder_forward(x)
        # ===================inference====================
        if is_training is False:
            saver = tf.train.Saver()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            if os.listdir(os.getcwd() + os.path.sep + 'neural_work' + os.path.sep + \
                          'checkpointfile'):
                saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(
                    os.getcwd() + os.path.sep + 'neural_work' + os.path.sep + \
                    'checkpointfile'))

            self.init(sess=sess)
            signal = sess.run(encoder_opt)
            sess.close()
            print('去噪模块inference完毕')
            return signal
        # ===================training=====================

        self._loss = tf.reduce_mean(tf.square(encoder_opt - y))
        self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        gvs = self._optimizer.compute_gradients(loss=self._loss,
                                          var_list=list(self._encoder_weights.values())
                                                   +list(self._decoder_weights.values()))
        opt_update = self._optimizer.apply_gradients(grads_and_vars=gvs)
        # opt_update = self._optimizer.minimize(loss=self._loss)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if os.listdir(os.getcwd() + os.path.sep + 'neural_work' + os.path.sep +\
                            'checkpointfile'):
                saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(os.getcwd() + os.path.sep + 'neural_work' + os.path.sep +\
                            'checkpointfile'))

            self.init(sess=sess)
            i = 1
            loss_optim = 1e9
            total_loss = 0
            while True:
                try:
                    _, loss = sess.run(fetches=[opt_update, self._loss])
                    total_loss += loss
                    curloss = total_loss / (i + 1)
                    if i % 100 == 0:
                        print(curloss)
                    i += 1
                    if curloss < loss_optim:
                        loss_optim = curloss
                        saver.save(sess=sess, save_path= \
                            os.getcwd() + os.path.sep + 'neural_work' + os.path.sep +\
                            'checkpointfile' + os.path.sep + 'save_model',
                                   write_meta_graph=True)
                except tf.errors.OutOfRangeError:
                    break


if __name__ == '__main__':
    #内建去噪数据
    sg = Signal(dataset_size=100, feature_size=20)
    dataset = np.hstack((sg.dataset, sg.dataset))
    # # print(dataset.shape)
    pq = ProcessQuzao(data=sg.dataset, epoch=500000)
    # pq.train()

    #正弦数据未进行预训练
    from sindata_example import SinusoidGenerator, generate_dataset
    s = SinusoidGenerator(K=20)
    train_ds, test_ds = generate_dataset(K=20)
    x, y = np.split(train_ds, 2, axis=-1)
    x = x.reshape(20000, -1)
    y = y.reshape(20000, -1)
    train_ds_Dataset = tf.data.Dataset.from_tensor_slices(tensors=(x, y)).repeat(50000).batch(512)
    pq.Dataset = train_ds_Dataset
    pq.train_and_predict(input_size=20)

