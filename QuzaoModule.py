#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: QuzaoModule
@time: 2021/6/14 4:59 下午
'''
import numpy as np
import pandas as pd
from EMCD import EMDblock
from xiaobo import Waveleting, Rsnr, Snr, Nsr, one_Er
from neural_work.processQuzao import ProcessQuzao
from neural_work.meta_learning_pretrain import Meta_process

class Quzao:
    def __init__(self, signal:np.ndarray, imfs_num:int=-1, use_meta_learn:bool=False):
        self._signal = signal
        self._imfs_and_res = EMDblock(signal=self._signal, imfs_num=imfs_num)()
        self._imfs = self._imfs_and_res[:-1]
        self._res = self._imfs_and_res[-1]
        self._after_quzao_imfs = np.sum([Waveleting(signal=imf)(6) for imf in self._imfs], axis=0) #高频部分
        self._after_quzao_res = ProcessQuzao(data=self._res, epoch=1)\
            .train_and_predict(input_size=self._res.shape[0], is_training=False)
        self._after_quzao_signal = self._after_quzao_imfs + self._after_quzao_res


    @property
    def after_quzao_signal(self):
        return self._after_quzao_signal


    #TODO 用去噪效果评估指标进行评估
    #TODO 探究小波去噪迭代次数对去噪效果影响
    #TODO 探究去噪自编码器中参数对去噪效果的影响



if __name__ == '__main__':
    import os
    print(os.getcwd() + os.path.sep)
