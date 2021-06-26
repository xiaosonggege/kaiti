#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: plotting
@time: 2021/6/26 10:17 上午
'''
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import xlrd

import re
import os

def gan_loss(pattern: re.Pattern=re.compile(pattern='\d+\.\d+$')):
    gan_loss = np.array([])
    with open(file=os.getcwd() + os.path.sep + 'gan_epoch1.txt') as f:
        while True:
            line_info = f.readline()
            if not line_info:
                break
            gan_loss = np.append(gan_loss, float(re.search(pattern=pattern, string=line_info).group(0)))

    print(gan_loss.shape)
    fig, ax = plt.subplots()
    ax.plot(gan_loss[:], c='b', label='Discriminator loss')

    ax.set_xlabel('Training epochs')
    ax.set_ylabel('loss function value')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    ax.legend(loc='upper right')
    fig.show()

def gan_loss2(pattern: re.Pattern=re.compile(pattern='\d+\.\d+$')):
    gan_loss = np.array([])
    with open(file=os.getcwd() + os.path.sep + 'gan_epoch2.txt') as f:
        while True:
            line_info = f.readline()
            if not line_info:
                break
            gan_loss = np.append(gan_loss, float(re.search(pattern=pattern, string=line_info).group(0)))

    print(gan_loss.shape)
    fig, ax = plt.subplots()
    ax.plot(gan_loss[:], c='g', label='Discriminator loss')

    ax.set_xlabel('Training epochs')
    ax.set_ylabel('loss function value')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    ax.legend(loc='upper right')
    fig.show()

def gan_generator_loss1(pattern: re.Pattern=re.compile(pattern='\d+\.\d+$')):
    gan_loss = np.array([])
    with open(file=os.getcwd() + os.path.sep + 'gan_generator_epoch1.txt') as f:
        while True:
            line_info = f.readline()
            if not line_info:
                break
            gan_loss = np.append(gan_loss, float(re.search(pattern=pattern, string=line_info).group(0)))

    print(gan_loss.shape)
    fig, ax = plt.subplots()
    ax.plot(gan_loss[:], c='b', label='Generator loss')

    ax.set_xlabel('Training epochs')
    ax.set_ylabel('loss function value')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    ax.legend(loc='upper left')
    fig.show()

def gan_generator_loss2(pattern: re.Pattern=re.compile(pattern='\d+\.\d+$')):
    gan_loss = np.array([])
    with open(file=os.getcwd() + os.path.sep + 'gan_generator_epoch2.txt') as f:
        while True:
            line_info = f.readline()
            if not line_info:
                break
            gan_loss = np.append(gan_loss, float(re.search(pattern=pattern, string=line_info).group(0)))

    print(gan_loss.shape)
    fig, ax = plt.subplots()
    ax.plot(gan_loss[:], c='g', label='Generator loss')

    ax.set_xlabel('Training epochs')
    ax.set_ylabel('loss function value')
    ax.grid(axis='x', linestyle='-.')
    ax.grid(axis='y', linestyle='-.')
    ax.legend(loc='upper left')
    fig.show()

def finepeaks():
    Adaboost_tree_num = ['10', '40', '50']
    xgboost_tree_num = ['20', '32', '50']
    cls_accuracy = ['SVM', 'Adaboost', 'Xgboost']
    cls_precision = ['SVM', 'Adaboost', 'Xgboost']
    cls_recall = ['SVM', 'Adaboost', 'Xgboost']
    adaboost_tree_num_acc = np.array([0.8547, 0.9042, 0.9290])
    xgboost_tree_num_acc = np.array([0.8963, 0.9281, 0.9405])
    accuracy = np.array([0.799, 0.916, 0.939])
    precision = np.array([0.614, 0.863, 0.912])
    recall = np.array([0.686, 0.884, 0.915])

    # # adaboost tree num
    # picture = plt.figure('Adaboost tree num')
    # index = np.arange(3)
    # bar_width = 0.35
    # opacity = 0.5
    # plt.bar(index, adaboost_tree_num_acc*100, bar_width, alpha=opacity,
    #         color='g')
    # plt.xticks(index, Adaboost_tree_num)
    # # plt.legend()
    # plt.grid(linestyle='--')
    # plt.ylim(0, 120)
    # for a, b in zip(index-0.15, adaboost_tree_num_acc*100):
    #     plt.text(a, b + 1.5, '%.2f%%' % b)
    # plt.ylabel('accuracy')
    # plt.xlabel('adaboost tree number')
    # picture.show()

    # xgboost tree num
    # picture = plt.figure('xgboost tree num')
    # index = np.arange(3)
    # bar_width = 0.35
    # opacity = 0.5
    # plt.bar(index, xgboost_tree_num_acc*100, bar_width, alpha=opacity,
    #         color='g')
    # plt.xticks(index, xgboost_tree_num)
    # # plt.legend()
    # plt.grid(linestyle='--')
    # plt.ylim(0, 120)
    # for a, b in zip(index-0.15, xgboost_tree_num_acc*100):
    #     plt.text(a, b + 1.5, '%.2f%%' % b)
    # plt.ylabel('accuracy')
    # plt.xlabel('xgboost tree number')
    # picture.show()

    # #accuracy
    # picture = plt.figure('accuracy of three classifier')
    # index = np.arange(3)
    # bar_width = 0.35
    # opacity = 0.5
    # plt.bar(index, accuracy * 100, bar_width, alpha=opacity,
    #         color='g')
    # plt.xticks(index, cls_accuracy)
    # # plt.legend()
    # plt.grid(linestyle='--')
    # plt.ylim(0, 120)
    # for a, b in zip(index - 0.15, accuracy * 100):
    #     plt.text(a, b + 1.5, '%.2f%%' % b)
    # plt.ylabel('accuracy')
    # plt.xlabel('classifier')
    # picture.show()

    # # precision
    # picture = plt.figure('precision of three classifier')
    # index = np.arange(3)
    # bar_width = 0.35
    # opacity = 0.5
    # plt.bar(index, precision * 100, bar_width, alpha=opacity,
    #         color='g')
    # plt.xticks(index, cls_precision)
    # # plt.legend()
    # plt.grid(linestyle='--')
    # plt.ylim(0, 120)
    # for a, b in zip(index - 0.15, precision * 100):
    #     plt.text(a, b + 1.5, '%.2f%%' % b)
    # plt.ylabel('precision')
    # plt.xlabel('classifier')
    # picture.show()

    # # recall
    # picture = plt.figure('recall of three classifier')
    # index = np.arange(3)
    # bar_width = 0.35
    # opacity = 0.5
    # plt.bar(index, recall * 100, bar_width, alpha=opacity,
    #         color='g')
    # plt.xticks(index, cls_recall)
    # # plt.legend()
    # plt.grid(linestyle='--')
    # plt.ylim(0, 120)
    # for a, b in zip(index - 0.15, recall * 100):
    #     plt.text(a, b + 1.5, '%.2f%%' % b)
    # plt.ylabel('recall')
    # plt.xlabel('classifier')
    # picture.show()

if __name__ == '__main__':
    # finepeaks()
    # gan_loss()
    # gan_loss2()
    # gan_generator_loss1()
    gan_generator_loss2()