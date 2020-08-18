#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: xiaobo
@time: 2020/8/17 7:45 下午
'''
import numpy as np
import pywt
from matplotlib import pyplot as plt

class Signal:
    def __init__(self, A, miuu, omiga, delta, length, signal_range):
        self._A = A
        self._miuu = miuu
        self._omiga = omiga
        self._delta = delta
        self._length = length
        self._signal_range = signal_range
        # 信号模拟
        self._signal = np.array([self._f_combine(n)[0] for n in self._signal_range])
        self._signal_sn = np.array([self._f_combine(n)[-1] for n in self._signal_range])

    @property
    def signal(self):
        return self._signal

    @property
    def signal_sn(self):
        return self._signal_sn

    def _f_combine(self, x):
        signal_single = 0
        f = lambda x, high, miu, sigma: (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-1 * ((x - miu) ** 2) / sigma)
        for high, miu, sigma in zip(self._A, self._miuu, self._omiga):
            signal_single += f(x, high, miu, sigma)
        signal_sn = signal_single
        signal_single *= (2 + self._delta * np.random.normal(loc=0, scale=1))
        return signal_single, signal_sn

class Waveleting:
    def __init__(self, signal:np.ndarray, wavelet_func:str='db8', threshold:np.float=0.04):
        self._signal = signal
        self._w = pywt.Wavelet(wavelet_func)
        self._threshold = threshold
        self._wavelet_func = wavelet_func
        self._maxlev = pywt.dwt_max_level(len(self._signal), self._w.dec_len)

    def __call__(self, L:int):
        signal = self.quzao_single()
        for i in range(L-1):
            signal = self.quzao_single(signal=signal)
        return signal

    def quzao_single(self, signal=None):
        data = self._signal if signal is None else signal
        self._coeffs = pywt.wavedec(data=data, wavelet=self._wavelet_func, level=self._maxlev)
        for i in range(1, len(self._coeffs)):
            self._coeffs[i] = pywt.threshold(data=self._coeffs[i], value=self._threshold * max(self._coeffs[i]))
        self._datarec = pywt.waverec(self._coeffs, 'db8')
        return self._datarec

    def heursure(self):
        N = self._signal.__len__()
        # 固定阈值
        sqtwolog = np.sqrt(2 * np.log(N))

        # 启发式阈值
        crit = np.sqrt((1 / N) * (np.log(N) / np.log(2.)) ** 3)
        eta = (np.sum(self._signal ** 2) - N) / N
        return sqtwolog if eta < crit else min(sqtwolog, crit)

def Rsnr(f_m:np.ndarray, f_n:np.ndarray):
    return np.sum((f_m - f_n) ** 2) / np.sum(f_m ** 2)

def Snr(f_m:np.ndarray, f_n:np.ndarray):
    return np.sum(f_m ** 2) / np.sum((f_m - f_n) ** 2)

def Nsr(f_m:np.ndarray, f_n:np.ndarray):
    return 1 / Snr(f_m=f_m, f_n=f_n)

def one_Er(f_m:np.ndarray, f_n:np.ndarray):
    return 1 - np.sum(f_m ** 2) / np.sum(f_n ** 2)

if __name__ == '__main__':
    # 信号模拟
    A = [5, 4, 2, 1, 8, 5, 5, 7, 1, 1, 10, 3, 2]
    miuu = [-90, -83, -65, -42, -40, -30, -10, 10, 20, 40, 50, 70, 90]
    omiga = [100, 40, 50, 40, 70, 80, 100, 60, 40, 50, 80, 50, 60]
    delta = 0.2
    length = 2000
    signal_range = np.linspace(-100, 100, length)
    signal = Signal(A=A, miuu=miuu, omiga=omiga, delta=delta, length=length, signal_range=signal_range)
    signal_raw = signal.signal
    signal_sn = signal.signal_sn

    # 原始信号绘图
    # fig, ax = plt.subplots()
    # ax.plot(np.linspace(-100, 100, length), signal_raw)
    # fig.show()

    #小波迭代降噪
    fig1, ax1 = plt.subplots()
    ax1.plot(np.linspace(-100, 100, length), signal_raw)

    f_m = signal_raw
    RSNR = []
    SNR = []
    NSR = []
    ONE_ER = []
    for i in range(1, 25): #14不错
        signal = Waveleting(signal=signal_raw)(i)
        f_n = signal
        RSNR.append(Rsnr(f_m=f_m, f_n=f_n))
        f_m = signal
        ax1.plot(np.linspace(-100, 100, length), signal+i*0.02)
    print(RSNR)
    # fig1.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(RSNR)
    fig2.show()