#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: GAN_signal
@time: 2021/6/19 10:21 下午
'''
import os
import ast
import json
from types import SimpleNamespace
import tensorflow as tf

FLAGS = None


def input_fn(path):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.apply(
        tf.data.experimental.shuffle_and_repeat(FLAGS.buffer_size, None))
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(_parse_fn, FLAGS.batch_size, num_parallel_batches=os.cpu_count()))
    dataset = dataset.prefetch(FLAGS.n_prefetch)
    return dataset


def _parse_fn(example):
    image, label = _decode(example)
    image = _normalize(image)
    image = _transform(image)
    return image, label


def _decode(example):
    features = tf.parse_single_example(
        example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)})
    image = tf.image.decode_jpeg(features['image'], channels=1)
    label = tf.cast(features['label'], tf.int32)
    return image, label


def _transform(image):
    image = tf.image.random_brightness(image, 0.5)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if FLAGS.channels_first:
        image = tf.transpose(image, [2, 0, 1])
    return image


def _normalize(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image - 0.5
    return image


def get_projection_input(images):
    if FLAGS.channels_first:
        images = tf.transpose(images, [0, 2, 3, 1])
    # pad to 3 channels
    images = tf.pad(images, [[0, 0], [0, 0], [0, 0], [1, 1]])
    images = tf.image.resize_images(images, FLAGS.projection_size)
    return images


def get_projection(images):
    projection_input = get_projection_input(images)
    incepv3 = tf.keras.applications.InceptionV3(
        weights='imagenet', include_top=False, input_tensor=projection_input, pooling='avg')
    projection = incepv3.output
    return projection


def model_fn(features, labels, mode, params):
    # projection = get_projection(features)
    # pseudo_label = tf.argmax(projection, -1)

    g_labels = tf.random_uniform((FLAGS.batch_size,), maxval=FLAGS.n_class, dtype=tf.int32, name='g_labels')

    noise = tf.random_normal((FLAGS.batch_size, FLAGS.z_shape), name='noise')
    valid = tf.ones((FLAGS.batch_size), tf.int32, 'valid')
    fake = tf.zeros((FLAGS.batch_size), tf.int32, 'fake')

    g_sample = generator(noise, g_labels, mode)
    d_logits_real, cls_logits_real = discriminator(features, mode)
    d_logits_fake, cls_logits_fake = discriminator(g_sample, mode, reuse=True)

    predictions = tf.argmax(cls_logits_real, -1, 'predictions', output_type=tf.int32)

    # PREDICTION
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    g_loss = tf.add_n([
        tf.losses.sparse_softmax_cross_entropy(valid, d_logits_fake, scope='losses/g_validity'),
        tf.losses.sparse_softmax_cross_entropy(g_labels, cls_logits_fake, scope='losses/g_cls')],
        'g_loss')
    d_loss = tf.add_n([
        tf.losses.sparse_softmax_cross_entropy(valid, d_logits_real, scope='losses/d_validity'),
        tf.losses.sparse_softmax_cross_entropy(fake, d_logits_fake, scope='losses/d_g_validity'),
        tf.losses.sparse_softmax_cross_entropy(labels, cls_logits_real, scope='losses/d_cls')],
        'd_loss')
    loss = tf.add_n([g_loss, d_loss], 'losses/loss')

    if FLAGS.channels_first:
        g_sample = tf.transpose(g_sample, [0, 2, 3, 1])
    tf.summary.image('g_sample', g_sample)
    tf.summary.scalar('g_loss', g_loss)
    tf.summary.scalar('d_loss', d_loss)

    train_accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
    tf.summary.scalar('train_accuracy', train_accuracy)

    tf.print(labels, output_stream=tf.logging.info)
    tf.print(tf.shape(labels), output_stream=tf.logging.info)
    tf.print(predictions, output_stream=tf.logging.info)
    tf.print(tf.shape(labels), output_stream=tf.logging.info)

    accuracy = tf.metrics.accuracy(labels, predictions, name='accuracy')
    metrics = {
        "accuracy": accuracy
    }
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        g_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'GAN/generator')
        d_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'GAN/discriminator')
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, FLAGS.beta1)
        g_op = optimizer.minimize(g_loss, tf.train.get_global_step(), g_var)
        d_op = optimizer.minimize(d_loss, tf.train.get_global_step(), d_var)
        train_op = tf.group(g_op, d_op)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)

    # EVAL
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)


def _bn(x, mode):
    training = mode == tf.estimator.ModeKeys.TRAIN
    bn_axis = 1 if FLAGS.channels_first else -1
    return tf.layers.batch_normalization(x, bn_axis, FLAGS.momentum, training=training)


def generator(noise, labels, mode, reuse=False):
    def _deconv(x, filters):
        return tf.layers.conv2d_transpose(x, filters, 3, 2, 'same', FLAGS.data_format, tf.nn.relu)

    with tf.variable_scope('GAN/generator', reuse=reuse):
        embedding = tf.get_variable('embedding', (FLAGS.n_class, FLAGS.z_shape))
        embedded = tf.nn.embedding_lookup(embedding, labels)
        embedded = tf.layers.flatten(embedded)

        x = tf.multiply(noise, embedded)

        x = tf.layers.dense(x, 64 * 6 * 6, tf.nn.relu)
        x = tf.reshape(x, (-1, 64, 6, 6) if FLAGS.channels_first else (-1, 6, 6, 64))
        x = _bn(x, mode)
        x = _deconv(x, 128)  # 12x12
        x = _bn(x, mode)
        x = _deconv(x, 128)  # 24x24
        x = _bn(x, mode)
        x = tf.pad(
            x, [[0, 0], [0, 0], [0, 1], [0, 1]] if FLAGS.channels_first
            else [[0, 0], [0, 1], [0, 1], [0, 0]])  # 25x25
        x = _deconv(x, 128)  # 50x50
        x = _bn(x, mode)
        x = _deconv(x, 64)  # 100x100
        x = _bn(x, mode)
        x = tf.layers.conv2d(
            x, FLAGS.channels, 3, 1, 'same', FLAGS.data_format, name='g_sample')

        return x


def discriminator(image, mode, reuse=False):
    def _conv(x, filters):
        return tf.layers.conv2d(
            x, filters, 3, 2, 'same', FLAGS.data_format, activation=tf.nn.leaky_relu)

    def _dropout(x):
        training = mode == tf.estimator.ModeKeys.TRAIN
        return tf.layers.dropout(x, FLAGS.drop_rate, training=training)

    with tf.variable_scope('GAN/discriminator', reuse=reuse):
        x = _conv(image, 16)  # 50x50
        x = _dropout(x)
        x = _conv(x, 32)  # 25x25
        x = _dropout(x)
        x = _bn(x, mode)
        x = _conv(x, 64)  # 13x13
        x = _dropout(x)
        x = _bn(x, mode)
        x = _conv(x, 128)  # 7x7
        x = _dropout(x)
        x = _bn(x, mode)
        x = _conv(x, 256)  # 4x4
        x = _dropout(x)
        x = tf.reduce_mean(x, [2, 3] if FLAGS.channels_first else [1, 2])

        d_logits = tf.layers.dense(x, 2, name='d_logits')
        cls_logits = tf.layers.dense(x, FLAGS.n_class, name='cls_logits')

        return d_logits, cls_logits


def main(_):
    # distribute = tf.contrib.distribute.ParameterServerStrategy(FLAGS.gpus)
    config = tf.estimator.RunConfig(
        save_summary_steps=1000,
        log_step_count_steps=1000,
        # train_distribute=distribute,
        # eval_distribute=distribute
    )

    model = tf.estimator.Estimator(model_fn, FLAGS.model_dir, config)

    # train_spec = tf.estimator.TrainSpec(lambda: input_fn(FLAGS.train_data), max_steps=FLAGS.max_steps)
    # eval_spec = tf.estimator.EvalSpec(lambda: input_fn(FLAGS.valid_data))

    # tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    model.train(lambda: input_fn(FLAGS.train_data), max_steps=FLAGS.max_steps)
    model.evaluate(lambda: input_fn(FLAGS.train_data))


if __name__ == "__main__":
    with open('params.json', 'r') as f:
        params = json.load(f)

    FLAGS = SimpleNamespace(**params)

    with open('.git/HEAD', 'r') as f_head:
        git_head_pointer = f_head.read().strip().split(': ')[-1]
        with open('.git/' + git_head_pointer, 'r') as f_git_hash:
            git_hash = f_git_hash.read()[:7]
    FLAGS.model_dir += git_hash

    FLAGS.target_size = ast.literal_eval(FLAGS.target_size)
    FLAGS.projection_size = ast.literal_eval(FLAGS.projection_size)
    FLAGS.data_format = 'channels_first' if FLAGS.channels_first else 'channels_last'

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

if __name__ == '__main__':
    pass
