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

class ProcessQuzao:
    def __init__(self, data:np.ndarray):
        self._data_raw = data
        self._encoder_weights = dict()
        self._decoder_weights = dict()
        self._batch_size = 512
        self._Dataset = tf.data.Dataset.from_tensor_slices(tensors=self._data_raw)\
        .batch(batch_size=self._batch_size)\
        .repeat(5)
        self._iterator = self._Dataset.make_initializable_iterator()
        self._next_batch = self._iterator.get_next()

    def init(self, sess:tf.Session):
        self._iterator.initializer.run(session=sess)


    def config_encoder_weights(self, input_size:int, *layers):
        for out_size in layers:
            self._encoder_weights['w%s' % out_size] = tf.get_variable(
                shape=(input_size, out_size),
                name='w%s' % out_size
            )
            self._encoder_weights['b%s' % out_size] = tf.get_variable(
                shape=(out_size),
                name='w%s' % out_size
            )

    def config_decoder_weights(self, input_size:int, *layers):
        for out_size in layers:
            self._decoder_weights['w%s' % out_size] = tf.get_variable(
                shape=(input_size, out_size),
                name='w%s' % out_size
            )
            self._decoder_weights['b%s' % out_size] = tf.get_variable(
                shape=(out_size),
                name='w%s' % out_size
            )

    def encoder_forward(self, data_batch:tf.Tensor):
        '''

        :return:
        '''
        self._encoder = None
        for i in self._encoder_weights:
            if self._encoder is None:
                self._encoder = tf.matmul(data_batch, self._encoder_weights['w%s' % i]) + self._encoder_weights['b%s' % i]
            else:
                self._encoder = tf.matmul(self._encoder, self._encoder_weights['w%s' % i]) + self._encoder_weights['b%s' % i]


    def decoder_forward(self, input:tf.Tensor):
        self._decoder = None
        for i in self._decoder_weights:
            if self._decoder is None:
                self._decoder = tf.matmul(input, self._decoder_weights['w%s' % i]) + self._decoder_weights['b%s' % i]
            else:
                self._decoder = tf.matmul(self._decoder, self._decoder_weights['w%s' % i]) + self._decoder_weights['b%s' % i]


    def train(self, data_batch:tf.Tensor, y_batch:tf.Tensor):
        #初始化
        encoder_layers = (1, 2, 3)
        self.config_encoder_weights(self._batch_size, *encoder_layers)
        decoder_layers = (3, 2, 1)
        self.config_decoder_weights(encoder_layers[-1], *decoder_layers)
        self.encoder_forward(self._next_batch)
        self.decoder_forward(self._encoder)
        self._loss = tf.square(self._decoder-y_batch)
        self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        gvs = self._optimizer.compute_gradients(loss=self._loss,
                                          var_list=list(self._encoder_weights.values())
                                                   +list(self._decoder_weights.values()))
        opt_update = self._optimizer.apply_gradients(grads_and_vars=gvs)
        with tf.Session() as sess:
            sess.run(tf.initialize_variables())
            self.init(sess=sess)
            for epoch in range(1000):
                _, loss = sess.run(fetches=[opt_update, self._loss])
                if (epoch % 10):
                    print(loss)




if __name__ == '__main__':
    a = np.array([1, 2, 3, 4])
    a = tf.data.Dataset.from_tensor_slices(a).repeat(4).batch(2)
    b = a.make_initializable_iterator()
    c = b.get_next()
    sess = tf.Session()
    b.initializer.run(session=sess)
    f = tf.constant(value=np.array([2, 3])) * c
    print(c.get_shape().as_list())
    def fun(x:tf.Tensor):
        return tf.constant(value=np.array([2, 3])) * x
    while (1):
        try:
            e = sess.run(fun(c))
        except tf.errors.OutOfRangeError:
            break
        else:
            print(e)
