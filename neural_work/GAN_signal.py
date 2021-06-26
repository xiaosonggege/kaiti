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
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from scipy.stats import norm #scipy 数值计算库
# import seaborn as sns #数据模块可视化
# import argparse #解析命令行参数
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# sns.set(color_codes=True) #设置主题颜色
# seed=42
# #设置随机数，使得每次生成的随机数相同
# np.random.seed(seed)
# tf.set_random_seed(seed)
#
# #define Gauss distribution:mean=4, std=0.5
# class DataDistribution(object):
#     def __init__(self):
#         self.mu= 4
#         self.sigma=0.5
#     #define sampling function
#     def samples(self,N):
#         #generate the number of n samples that mean=mu,std=sigama
#         samples=np.random.normal(self.mu,self.sigma,N)
#         samples.sort()
#         return samples
#
# #define a liner computation function
# #args input:the inputting samples, output_dim:the dimension of output
# #scope:variable space, stddev:std
# #the liner function is to compute y=wx+b
# def linear(input, output_dim, scope=None, stddev=1.0):
#     #initialize the norm randoms
#     norm = tf.random_normal_initializer(stddev=stddev)
#     #initialize the const
#     const= tf.constant_initializer(0.0)
#     #computet the y=wx+b
#     #open the variable space named arg scope or 'linear'
#     with tf.variable_scope(scope or 'linear'):
#         #get existed variable named 'w' whose shape is defined as [input,get_shape()[1],output_dim]
#         #and use norm or const distribution to initialize the tensor
#         w = tf.get_variable('w',[input.get_shape()[1],output_dim],initializer=norm)
#         b= tf.get_variable('b', [output_dim],initializer=const)
#         return tf.matmul(input,w)+b
#
# #define the noise distribution
# #use linear space to split -range to range into N parts plus random noise
# class GeneratorDistribution():
#     def __init__(self,range):
#         self.range=range
#     def samples(self,N):
#         return np.linspace(-self.range,self.range,N)+np.random.random(N)*0.01
#
# #define generator nets using soft-plus function
# #whose nets have only one hidden layer one input layer
# def generator(input, hidden_size):
#     #soft-plus function:log(exp(features)+1)
#     #h0 represents the output of the input layer
#     h0=tf.nn.softplus(linear(input, hidden_size),'g0')
#     #the output dimension is 1
#     h1=linear(h0,1,'g1')
#     return h1
#
# #define the discriminator nets using deep tanh function
# #because the discriminator nets usually have the stronger learning abilitiy
# #to train the better generator
# def discriminator(input, hidden_size,minibatch_layer=True):
#     #the output dimension is 2 multiply hidden_size because its need to be deep
#     h0=tf.tanh(linear(input,hidden_size*2,'d0'))
#     h1=tf.tanh(linear(h0,hidden_size*2,'d1'))
#     if minibatch_layer:
#         h2=minibatch(h1)
#     else:
#         h2=tf.tanh(linear(h1, hidden_size*2,'d2'))
#     h3 = tf.sigmoid(linear(h2, 1, 'd3'))
#     return h3
#
# def minibatch(input, num_kernels=5, kernel_dim=3):
#     x=linear(input, num_kernels*kernel_dim,scope='minibatch',stddev=0.02)
#     activation=tf.reshape(x,(-1, num_kernels,kernel_dim))
#     diffs=tf.expand_dims(activation,3)-tf.expand_dims(tf.transpose(activation,[1,2,0]),0)
#     abs_diffs=tf.reduce_sum(tf.abs(diffs),2)
#     minibatch_features=tf.reduce_sum(tf.exp(-abs_diffs),2)
#     return tf.concat([input, minibatch_features],1)
#
# #define optimizer
# #using decay learning and GradientDescentOptimizer
# def optimizer(loss, val_list,initial_learning_rate=0.005):
#     decay=0.95 #the speed of decline
#     num_decay_steps= 150 #for every 150 steps learning rate decline
#     batch=tf.Variable(0)
#     learning_rate=tf.train.exponential_decay(
#         initial_learning_rate,
#         batch,
#         num_decay_steps,
#         decay,
#         staircase=True
#     )
#     optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(
#         loss,
#         global_step=batch,
#         var_list=val_list
#     )
#     return optimizer
#
# class GAN(object):
#     def __init__(self,data,gen,num_steps,batch_size,minibatch,log_every,anim_path):
#         self.data=data
#         self.gen=gen
#         self.num_steps=num_steps
#         self.batch_size=batch_size
#         self.minibatch=minibatch
#         self.log_every=log_every
#         self.mlp_hidden_size=4
#         self.anim_path=anim_path
#         self.anim_frames=[]
#
#         #if using minibatch then decline the learning rate
#         #or improve the learning rate
#         if self.minibatch:
#             self.learning_rate=0.005
#         else:
#             self.learning_rate=0.03
#
#         self._create_model()
#
#     def _create_model(self):
#         #in order to make sure that the discriminator is  providing useful gradient
#         #imformation,we are going to pretrain the discriminator using a maximum
#         #likehood objective,we define the network for this pretraining step scoped as D_pre
#         with tf.variable_scope('D_pre'):
#             self.pre_input=tf.placeholder(tf.float32,shape=[self.batch_size,1])
#             self.pre_labels=tf.placeholder(tf.float32,shape=[self.batch_size,1])
#             D_pre=discriminator(self.pre_input, self.mlp_hidden_size,self.minibatch)
#             self.pre_loss=tf.reduce_mean(tf.square(D_pre-self.pre_labels))
#             self.pre_opt=optimizer(self.pre_loss,None,self.learning_rate,)
#
#         #this defines the generator network:
#         #it takes samples from a noise distribution
#         #as input, and passes them through an MLP
#         with tf.variable_scope('Gen'):
#             self.z=tf.placeholder(tf.float32,[self.batch_size,1])
#             self.G=generator(self.z,self.mlp_hidden_size)
#
#         #this discriminator tries to tell the difference between samples
#         #from  the true
#         #x is the real sample while z is the generated samples
#         with tf.variable_scope('Disc') as scope:
#             self.x=tf.placeholder(tf.float32,[self.batch_size,1])
#             self.D1=discriminator(self.x,self.mlp_hidden_size,self.minibatch)
#             scope.reuse_variables()
#             self.D2=discriminator(self.G,self.mlp_hidden_size,self.minibatch)
#
#         #define the loss for discriminator and generator network
#         #and create optimizer for both
#         self.loss_d=tf.reduce_mean(-tf.log(self.D1)-tf.log(1-self.D2))
#         self.loss_g=tf.reduce_mean(-tf.log(self.D2))
#
#         self.d_pre_params=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'D_pre')
#         self.d_params=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'Disc')
#         self.g_params=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'Gen')
#
#         self.opt_d=optimizer(self.loss_d,self.d_params,self.learning_rate)
#         self.opt_g=optimizer(self.loss_g,self.g_params,self.learning_rate)
#
#     def train(self):
#         with tf.Session() as session:
#             tf.global_variables_initializer().run()
#             #pretraining discriminator
#             num_pretraining_steps=1000
#             for step in range(num_pretraining_steps):
#                 d=(np.random.random(self.batch_size)-0.5)*10.0
#                 labels=norm.pdf(d,loc=self.data.mu,scale=self.data.sigma)
#                 pretrain_loss,_=session.run(
#                     [self.pre_loss,self.pre_opt],
#                     {
#                         self.pre_input:np.reshape(d,(self.batch_size,1)),
#                         self.pre_labels:np.reshape(labels,(self.batch_size,1))
#                     }
#                 )
#             self.weightsD=session.run(self.d_pre_params)
#
#             #copy weights from pretraining over to new D network
#             for i,v in enumerate(self.d_params):
#                 session.run(v.assign(self.weightsD[i]))
#
#             for step in range(self.num_steps):
#                 #update discriminator
#                 x=self.data.samples(self.batch_size)
#                 z=self.gen.samples(self.batch_size)
#                 loss_d,_=session.run(
#                     [self.loss_d,self.opt_d],
#                     {
#                         self.x:np.reshape(x,(self.batch_size,1)),
#                         self.z:np.reshape(z,(self.batch_size,1))
#                     }
#                 )
#
#                 #update generator
#                 z=self.gen.samples(self.batch_size)
#                 loss_g,_=session.run(
#                     [self.loss_g,self.opt_g],
#                     {
#                         self.z:np.reshape(z,(self.batch_size,1))
#                     }
#                 )
#
#                 if step % self.log_every==0:
#                     print('{}:{}\t{}'.format(step,loss_d,loss_g))
#                 if self.anim_path:
#                     self.anim_frames.append(self._samples(session))
#
#             if self.anim_path:
#                 self._save_animation()
#             else:
#                 self._plot_distributions(session)
#
#     def _samples(self, session, num_points=10000, num_bins=100):
#         # return a tuple (db,pd,pg), where db is the current decision boundary
#         # pd is a histogram of samples from the data distribution,
#         # and pg is a histogram of generated samples.
#         xs = np.linspace(-self.gen.range, self.gen.range, num_points)
#         bins = np.linspace(-self.gen.range, self.gen.range, num_bins)
#
#         # decision boundary
#         db = np.zeros((num_points, 1))
#         for i in range(num_points // self.batch_size):
#             db[self.batch_size * i:self.batch_size * (i + 1)] = session.run(
#                 self.D1,
#                 {
#                     self.x: np.reshape(
#                         xs[self.batch_size * i:self.batch_size * (i + 1)],
#                         (self.batch_size, 1)
#                     )
#                 }
#             )
#         # data distribution
#         d = self.data.samples(num_points)
#         pd, _ = np.histogram(d, bins=bins, density=True)
#
#         # generated samples
#         zs = np.linspace(-self.gen.range, self.gen.range, num_points)
#         g = np.zeros((num_points, 1))
#         # // 整数除法
#         for i in range(num_points // self.batch_size):
#             g[self.batch_size * i:self.batch_size * (i + 1)] = session.run(
#                 self.G,
#                 {
#                     self.z: np.reshape(
#                         zs[self.batch_size * i: self.batch_size * (i + 1)],
#                         (self.batch_size, 1)
#                     )
#                 }
#             )
#         pg, _ = np.histogram(g, bins=bins, density=True)
#         return db, pd, pg
#
#     def _plot_distributions(self, session):
#         db, pd, pg = self._samples(session)
#         db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
#         p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
#         f, ax = plt.subplots(1)
#         ax.plot(db_x, db, label='decision boundary')
#         ax.set_ylim(0, 1)
#         plt.plot(p_x, pd, label='real data')
#         plt.plot(p_x, pg, label='generated data')
#         plt.title('1D Generative Adversarial Network')
#         plt.xlabel('Data values')
#         plt.ylabel('Probability density')
#         plt.legend()
#         plt.show()
#
#     def _save_animation(self):
#         f, ax = plt.subplots(figsize=(6, 4))
#         f.suptitle('1D Generative Adversarial Network', fontsize=15)
#         plt.xlabel('Data values')
#         plt.ylabel('Probability density')
#         ax.set_xlim(-6, 6)
#         ax.set_ylim(0, 1.4)
#         line_db, = ax.plot([], [], label='decision boundary')
#         line_pd, = ax.plot([], [], label='real data')
#         line_pg, = ax.plot([], [], label='generated data')
#         frame_number = ax.text(
#             0.02,
#             0.95,
#             '',
#             horizontalalignment='left',
#             verticalalignment='top',
#             transform=ax.transAxes
#         )
#         ax.legend()
#
#         db, pd, _ = self.anim_frames[0]
#         db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
#         p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
#
#         def init():
#             line_db.set_data([], [])
#             line_pd.set_data([], [])
#             line_pg.set_data([], [])
#             frame_number.set_text('')
#             return (line_db, line_pd, line_pg, frame_number)
#
#         def animate(i):
#             frame_number.set_text(
#                 'Frame: {}/{}'.format(i, len(self.anim_frames))
#             )
#             db, pd, pg = self.anim_frames[i]
#             line_db.set_data(db_x, db)
#             line_pd.set_data(p_x, pd)
#             line_pg.set_data(p_x, pg)
#             return (line_db, line_pd, line_pg, frame_number)
#
#         anim = animation.FuncAnimation(
#             f,
#             animate,
#             init_func=init,
#             frames=len(self.anim_frames),
#             blit=True
#         )
#         anim.save(self.anim_path, fps=30, extra_args=['-vcodec', 'libx264'])
#
#
# def main(args):
#     model = GAN(
#         DataDistribution(),
#         GeneratorDistribution(range=8),
#         args.num_steps,
#         args.batch_size,
#         args.minibatch,
#         args.log_every,
#         args.anim
#     )
#     model.train()
#
#
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num-steps', type=int, default=1200,
#                         help='the number of training steps to take')
#     parser.add_argument('--batch-size', type=int, default=12,
#                         help='the batch size')
#     parser.add_argument('--minibatch', type=bool, default=False,
#                         help='use minibatch discrimination')
#     parser.add_argument('--log-every', type=int, default=10,
#                         help='print loss after this many steps')
#     parser.add_argument('-anim', type=str, default=None,
#                         help='the name of the output animation file (default: none)')
#     return parser.parse_args()
#
#
# if __name__ == '__main__':
#     main(parse_args())

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    tf.set_random_seed(1)
    np.random.seed(1)

    BATCH_SIZE = 100
    LR_G = 0.0001
    LR_D = 0.0001
    GEN_DIM = 5
    INPUT_DIM = 10

    with tf.variable_scope('Generator'):
        # 生成器，用来伪造数据
        is_train = tf.placeholder(tf.int32)
        # if is_train == 1:
        G_in = tf.placeholder(tf.float32, [None, GEN_DIM])  # shape = (64,5)
        G_l1 = tf.layers.dense(G_in, 256, tf.nn.relu)
        # G_l2 = tf.layers.dense(G_l1, 128, tf.nn.relu)
        G_out = tf.layers.dense(G_l1, INPUT_DIM)

    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        # 判别器
        real_data = tf.placeholder(tf.float32, [None, INPUT_DIM], name='real_data')
        real_output = tf.layers.dense(real_data, 128, tf.nn.relu, name='l1')
        real_output = tf.layers.dense(real_output, 1, tf.nn.sigmoid, name='out')

        # fake art
        fake_output = tf.layers.dense(G_out, 128, tf.nn.relu, name='1')
        fake_output = tf.layers.dense(fake_output, 1, tf.nn.sigmoid, name='out')

    D_loss = -tf.reduce_mean(tf.log(real_output) + tf.log(1 - fake_output))  # 定义误差函数
    G_loss = -tf.reduce_mean(tf.log(fake_output))

    train_D = tf.train.AdamOptimizer(LR_D).minimize(  # 定义优化函数
                D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))

    train_G = tf.train.AdamOptimizer(LR_G).minimize(
                G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

    sess = tf.Session()  # 初始化流图
    sess.run(tf.global_variables_initializer())
    from appling.basic_line import peak_with_boundary
    qujixian_result, peaks_start, peaks_end, peaks_with_boundary, x_a1 = peak_with_boundary()
    dataset = qujixian_result[:1000].reshape(-1, 10)
    # print(dataset.shape)
    for step in range(5000):
        G_ideas = np.random.randn(BATCH_SIZE, GEN_DIM)
        # print(G_ideas.shape)
        # G_paintings, pa0, D1 = sess.run([G_out, prob_artist0, D_loss, train_D, train_G],
        #                                 {G_in: G_ideas, real_art: artist_paintings})[:3]
        _, D_loss_ = sess.run(fetches=[train_D, D_loss], feed_dict={G_in:G_ideas, real_data:dataset})
        _, G_loss_ = sess.run(fetches=[train_G, G_loss], feed_dict={G_in:G_ideas})
        if step % 100 == 0:
            print('=============', G_loss_)
