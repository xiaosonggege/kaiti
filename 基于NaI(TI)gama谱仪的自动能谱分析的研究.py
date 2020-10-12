#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: 基于NaI(TI)gama谱仪的自动能谱分析的研究
@time: 2020/10/11 3:06 下午
'''
import numpy as np
import pandas as pd
import copy

class Property_s:
    def __init__(self, name):
        self._name = '_' + name

    def __get__(self, instance, owner):
        return instance.__dict__[self._name]

    def __set__(self, instance, value):
        instance.__dict__[self._name] = value

class SignalProcessing:
    '''
    数据平滑->对数变换->寻峰->计算峰值区域边界->本地面积->净峰面积
    '''
    @classmethod
    def signal_extand(cls, signal:np.ndarray, extand:int)->np.ndarray:
        '''
        在平滑处理前需要进行信号两段用0扩展
        :param signal:
        :param extand:
        :return:
        '''
        signal_extand = np.ones(shape=(extand, ), dtype=np.float)
        signal = np.hstack((signal_extand, signal, signal_extand))
        return signal

    @classmethod
    def pinghua(cls, signal:np.ndarray, weights:np.ndarray, k:np.float)->np.ndarray:
        '''
        多点平滑去噪
        :param signal:
        :param weights:
        :return:
        '''
        signal_ans = np.array([])
        for i in range(1, signal.shape[0]-weights.shape[0], 1):
            signal_ans = np.append(signal_ans, k * (weights @ signal[weights.shape[0]*(i-1): weights.shape[0]*(i-1)+(weights.shape[0]-1)]))
        return signal_ans

    @classmethod
    def loging(cls, signal:np.ndarray, A:np.float, B:np.float)->np.ndarray:
        return A * np.log10(B + signal)

    @classmethod
    def duichenlingmianjifa(cls, signal:np.ndarray, extand:int, thr:np.float)->list:
        def C_j(j:int)->np.float:
            return np.exp(-4 * (j / 4) ** 2 * np.log(2))
        signal_extand = SignalProcessing.signal_extand(signal=signal, extand=extand)
        c_j = np.array([C_j(j) for j in range(-extand, extand+1, 1)])
        #计算对称领面积变换、灵敏因子计算
        y = copy.deepcopy(signal_extand)
        delta_y = copy.deepcopy(signal_extand)
        for i in range(extand, signal_extand - extand + 1, 1):
            y[i] = c_j @ signal_extand[i-extand:i+extand]
            delta_y[i] = np.sqrt((c_j ** 2) @ signal_extand[i-extand:i+extand])
        ss_i = y[extand:-extand] / delta_y[extand:-extand]
        #求峰值点索引
        index = list(range(ss_i))
        fengzhi_index = np.array(index)[np.where(ss_i>thr)].tolist()
        #求峰值区域索引
        signal_extand2 = SignalProcessing.signal_extand(signal=signal, extand=2)
        signal_extand2_pinghua = np.array([1/12*(signal_extand2[i-2]-8*signal_extand2[i-1]+
                                          8*signal_extand2[i+1]-signal_extand2[i+2]) for i in range(2, -2)])
        #峰值索引序列中加入头尾索引
        fengzhi_index = np.insert(fengzhi_index, 0, 0)
        fengzhi_index = np.append(fengzhi_index, ss_i.shape[0]-1)
        left_range = []
        right_range = []
        for i, j in zip(fengzhi_index[:-1], fengzhi_index[0:]):
            #向左找
            go_left = j
            while signal_extand2_pinghua[go_left] >= 0:
                go_left -= 1
            left_range.append(go_left)
            #向右找
            go_right = i
            while signal_extand2_pinghua[go_right] <= 0:
                go_right += 1
            right_range.append(go_right)
        fengzhi_rigon = list(zip(left_range[:-1], right_range[1:]))
        return fengzhi_rigon

    @classmethod
    def bendi_calc(cls, signal:np.ndarray, fengzhi_rigon:list)->np.ndarray:
        signal = SignalProcessing.signal_extand(signal=signal, extand=2)
        signal = SignalProcessing.pinghua(signal=signal, weights=np.array([1, 4, 16, 4, 1]), k=1/16)
        bendi_s = np.array([])
        index = 1
        for i, j in fengzhi_rigon:
            lnB = ((np.log(signal[j]) - np.log(signal[i])) / (j - i)) * (index - i) + np.log(signal[i])
            bendi_s = np.append(bendi_s, np.exp(lnB))
            index += 1
        return bendi_s

    @classmethod
    def jingfeng_calc(cls, signal:np.ndarray, fengzhi_rigon:list, bendi_s:np.ndarray)->np.ndarray:
        jingfneg_s = np.array([])
        index = 0
        for i, j in fengzhi_rigon:
            jingfneg_s = np.append(jingfneg_s, np.sum(signal[i, j+1]) - bendi_s[index])
            index += 1
        return jingfneg_s


    def __init__(self, signal:np.ndarray)->None:
        self._signal = signal
    Signal = Property_s('signal')

    def processing(self)->tuple:
        #平滑
        signal_pinghua_ex = SignalProcessing.signal_extand(signal=self._signal, extand=2)
        signal_pinghua = SignalProcessing.pinghua(signal=signal_pinghua_ex,
                                                  weights=np.array([1, 4, 16, 4, 1], dtype=np.float), k=1/16)
        signal_pinghua_log = SignalProcessing.loging(signal=signal_pinghua, A=0.9, B=0.2)
        #寻峰
        fengzhi_rigon = SignalProcessing.duichenlingmianjifa(signal=signal_pinghua_log, extand=5, thr=0.5) #thr需要修改
        #本底计算
        bendi_s = SignalProcessing.bendi_calc(signal=signal_pinghua, fengzhi_rigon=fengzhi_rigon)
        #净峰面积
        jingfeng_s = SignalProcessing.jingfeng_calc(signal=signal_pinghua, fengzhi_rigon=fengzhi_rigon, bendi_s=bendi_s)
        return fengzhi_rigon, bendi_s, jingfeng_s

    def __enter__(self):
        return self.processing()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True

if __name__ == '__main__':
    signal = None
    with SignalProcessing(signal=signal) as s:
        fengzhi_rigon, bendi_s, jingfeng_s = s
        print(fengzhi_rigon)
        print(bendi_s)
        print(jingfeng_s)

