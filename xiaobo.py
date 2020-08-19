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
        signal_single *= (250 + self._delta * np.random.normal(loc=0, scale=1))
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
    delta = 25
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
    ax1.plot(np.linspace(-100, 100, length), signal_raw, label='L=1')

    f_m = signal_raw
    RSNR = []
    SNR = []
    NSR = []
    ONE_ER = []
    for i in range(1, 20): #14不错 #20
        signal = Waveleting(signal=signal_raw)(i)
        f_n = signal
        RSNR.append(Rsnr(f_m=f_m, f_n=f_n))
        SNR.append(10 * np.log2(Snr(f_m=signal_sn, f_n=f_n)))
        NSR.append(10 * np.log2(Nsr(f_m=signal_sn, f_n=f_n)))
        ONE_ER.append(one_Er(f_m=f_n, f_n=signal_sn))
        f_m = signal
    #     ax1.plot(np.linspace(-100, 100, length), signal+i*3, label='L='+str(i+2))
    # ax1.legend()
    # ax1.set_xlabel('Number of tracks')
    # ax1.set_ylabel('Counts')
    # fig1.show()

    fig2, ax2 = plt.subplots()
    # ax2.plot(RSNR, label='RSNR', marker='o', markersize=5)
    # ax2.plot(SNR, label='SNR', marker='^', markersize=5)
    # ax2.plot(NSR, label='NSR', marker='s', markersize=5)
    ax2.plot(ONE_ER, label='|1-ER|', marker='d', markersize=5)
    ax2.legend(loc=4)
    ax2.set_xlabel('Number of noise reduction iterations')
    ax2.set_ylabel('Index value')
    ax2.set_xticks([i for i in range(0, 19)])
    ax2.set_xticklabels([i for i in range(1, 20)])
    # fig2.show()

#########################
    f_m_1 = signal_raw
    f_m_2 = signal_raw
    RSNR_db8 = []
    RSNR_sym8 = []
    for i in range(1, 20): #14不错 #20
        signal_db8 = Waveleting(signal=signal_raw)(i)
        signal_sym8 = Waveleting(signal=signal_raw, wavelet_func='sym8')(i)
        f_n_1 = signal_db8
        f_n_2 = signal_sym8
        RSNR_db8.append(1000*Rsnr(f_m=f_m_1, f_n=f_n_1))
        RSNR_sym8.append(Rsnr(f_m=f_m_2, f_n=f_n_2))
        # SNR.append(10 * np.log2(Snr(f_m=signal_sn, f_n=f_n)))
        # NSR.append(10 * np.log2(Nsr(f_m=signal_sn, f_n=f_n)))
        # ONE_ER.append(one_Er(f_m=f_n, f_n=signal_sn))
        f_m_1 = signal_db8
        f_m_2 = signal_sym8
    fig2, ax2 = plt.subplots()
    ax2.plot(RSNR_db8, label='db8', marker='o', markersize=5)
    ax2.plot(RSNR_sym8, label='sym8', marker='d', markersize=5)
    ax2.legend(loc=1)
    ax2.set_xlabel('Number of noise reduction iterations')
    ax2.set_ylabel('Index value')
    ax2.set_xticks([i for i in range(0, 19)])
    ax2.set_xticklabels([i for i in range(1, 20)])
    # fig2.show()

#######################################
    RSNR_20 = []
    RSNR_50 = []
    RSNR_120 = []
    signal_20 = Signal(A=A, miuu=miuu, omiga=omiga, delta=delta*0.2, length=length, signal_range=signal_range)
    signal_50 = Signal(A=A, miuu=miuu, omiga=omiga, delta=delta * 0.5, length=length, signal_range=signal_range)
    signal_120 = Signal(A=A, miuu=miuu, omiga=omiga, delta=delta * 1.2, length=length, signal_range=signal_range)
    f_m_1 = signal_20.signal
    f_m_2 = signal_50.signal
    f_m_3 = signal_120.signal

    for i in range(1, 20): #14不错 #20
        signall_20 = Waveleting(signal=signal_20.signal)(i)
        signall_50 = Waveleting(signal=signal_50.signal)(i)
        signall_120 = Waveleting(signal=signal_120.signal)(i)
        f_n_1 = signall_20
        f_n_2 = signall_50
        f_n_3 = signall_120
        RSNR_20.append(1000*Rsnr(f_m=f_m_1, f_n=f_n_1))
        RSNR_50.append(1000*Rsnr(f_m=f_m_2, f_n=f_n_2))
        RSNR_120.append(1000*Rsnr(f_m=f_m_3, f_n=f_n_3))
        f_m_1 = signall_20
        f_m_2 = signall_50
        f_m_3 = signall_120
    fig2, ax2 = plt.subplots()
    ax2.plot(RSNR_20, label='20%', marker='o', markersize=5)
    ax2.plot(RSNR_50, label='50%', marker='d', markersize=5)
    ax2.plot(RSNR_120, label='120%', marker='s', markersize=5)
    ax2.legend(loc=1)
    ax2.set_xlabel('Number of noise reduction iterations')
    ax2.set_ylabel('Index value')
    ax2.set_xticks([i for i in range(0, 19)])
    ax2.set_xticklabels([i for i in range(1, 20)])
    fig2.show()
