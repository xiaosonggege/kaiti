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
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

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
            self._coeffs[i] = pywt.threshold(data=self._coeffs[i], value=self._threshold * max(self._coeffs[i])) #self._threshold * max(self._coeffs[i]) self.heursure()
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
    return np.sum(f_m ** 2) / np.sum(f_n ** 2)

if __name__ == '__main__':
    # 信号模拟
    # A = [5, 4, 2, 1, 8, 5, 5, 7, 1, 1, 10, 3, 2]
    # miuu = [-90, -83, -65, -42, -40, -30, -10, 10, 20, 40, 50, 70, 90]
    # omiga = [100, 40, 50, 40, 70, 80, 100, 60, 40, 50, 80, 50, 60]
    # delta = 25
    # length = 2000
    # signal_range = np.linspace(-100, 100, length)
    # signal = Signal(A=A, miuu=miuu, omiga=omiga, delta=delta, length=length, signal_range=signal_range)
    # signal_raw = signal.signal
    # signal_sn = signal.signal_sn
    #真实信号
    figure_a1 = pd.read_excel(io='/Users/songyunlong/Desktop/实验室/寿老师项目/1-a(1).xlsx', sheet_name='1-a', index_col=None, header=1)
    # figure_a2 = pd.read_excel(io='/Users/songyunlong/Desktop/1-b(1).xlsx', sheet_name='1-b', index_col=None, header=1)
    figure_data_a1 = figure_a1.values.T
    # figure_data_a2 = figure_a2.values.T

    # print(figure_data_a1.shape)
    # print(figure_data_a2.shape)
    x_a1, y_a1 = np.split(ary=figure_data_a1, indices_or_sections=2, axis=0)
    # 插值
    spline = lambda x, y: interp1d(x, y, kind='linear') #quadratic
    x_spline_new = np.linspace(np.min(x_a1), np.max(x_a1), 3000)
    # print(x_a1.shape, y_a1.shape)
    figure_data_a1_chayang = spline(x_a1.ravel(), y_a1.ravel())(x_spline_new)
    # x_a2, y_a2 = np.split(ary=figure_data_a2, indices_or_sections=2, axis=0)
    rng = np.random.RandomState(0)
    noise = rng.normal(loc=0, scale=1, size=figure_data_a1_chayang.shape)
    y_a1_addnoise = figure_data_a1_chayang + 0.5 * noise

    # 原始信号绘图
    # fig, ax = plt.subplots(nrows=2)
    # ax[0].plot(x_spline_new, figure_data_a1_chayang.ravel(), label='raw signal')
    # # ax[1].plot(x_a2.ravel(), y_a2.ravel())
    # ax[1].plot(x_spline_new, y_a1_addnoise.ravel(), label='add noise')
    # ax[0].legend()
    # ax[1].legend()
    # ax[0].set_xlabel('Number of noise reduction iterations')
    # ax[0].set_ylabel('Index value')
    # ax[1].set_xlabel('Number of noise reduction iterations')
    # ax[1].set_ylabel('Index value')
    # fig.show()

    #小波迭代降噪
    fig1, ax1 = plt.subplots()
    ax1.plot(x_spline_new, y_a1_addnoise.ravel(), label='L=1')

    f_m = y_a1_addnoise.ravel()
    RSNR = []
    SNR = []
    NSR = []
    ONE_ER = []
    for i in range(1, 16): #6不错 #16
        signal = Waveleting(signal=y_a1_addnoise.ravel())(i)
        f_n = signal
        RSNR.append(Rsnr(f_m=f_m, f_n=f_n))
        SNR.append(1e-4 * np.log2(Snr(f_m=figure_data_a1_chayang.ravel(), f_n=f_n)))
        NSR.append(1e-4 * np.log2(Nsr(f_m=figure_data_a1_chayang.ravel(), f_n=f_n)))
        ONE_ER.append(1e-4 * one_Er(f_m=f_n, f_n=figure_data_a1_chayang.ravel()))
        f_m = signal
    #     ax1.plot(x_spline_new, signal+i*2, label='L='+str(i+2))
    # ax1.legend()
    # ax1.set_xlabel('Number of tracks')
    # ax1.set_ylabel('Counts')
    # fig1.show()
    #
    fig2, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    print(type(ax3))
    ax1.plot(RSNR, label='RSNR', marker='o', markersize=5)
    ax2.plot(SNR, label='SNR', marker='^', markersize=5)
    ax3.plot(NSR, label='NSR', marker='s', markersize=5)
    ax4.plot(ONE_ER, label='|1-ER|', marker='d', markersize=5)
    ax1.legend(loc=1)
    ax2.legend(loc=4)
    ax3.legend(loc=1)
    ax4.legend(loc=1)
    ax1.set_xlabel('Number of noise reduction iterations')
    ax1.set_ylabel('Index value')
    ax1.set_xticks([i for i in range(0, 16)])
    ax1.set_xticklabels([i for i in range(0, 16)])
    ax2.set_xlabel('Number of noise reduction iterations')
    ax2.set_ylabel('Index value')
    ax2.set_xticks([i for i in range(0, 16)])
    ax2.set_xticklabels([i for i in range(0, 16)])
    ax3.set_xlabel('Number of noise reduction iterations')
    ax3.set_ylabel('Index value')
    ax3.set_xticks([i for i in range(0, 16)])
    ax3.set_xticklabels([i for i in range(0, 16)])
    ax4.set_xlabel('Number of noise reduction iterations')
    ax4.set_ylabel('Index value')
    ax4.set_xticks([i for i in range(0, 16)])
    ax4.set_xticklabels([i for i in range(0, 16)])
    fig2.show()

#########################
    # f_m_1 = y_a1_addnoise.ravel()
    # f_m_2 = y_a1_addnoise.ravel()
    # RSNR_db8 = []
    # RSNR_sym8 = []
    # for i in range(1, 20): #14不错 #20
    #     signal_db8 = Waveleting(signal=y_a1_addnoise.ravel())(i)
    #     signal_sym8 = Waveleting(signal=y_a1_addnoise.ravel(), wavelet_func='sym8')(i)
    #     f_n_1 = signal_db8
    #     f_n_2 = signal_sym8
    #     RSNR_db8.append(1000*Rsnr(f_m=f_m_1, f_n=f_n_1))
    #     RSNR_sym8.append(Rsnr(f_m=f_m_2, f_n=f_n_2))
    #     # SNR.append(10 * np.log2(Snr(f_m=signal_sn, f_n=f_n)))
    #     # NSR.append(10 * np.log2(Nsr(f_m=signal_sn, f_n=f_n)))
    #     # ONE_ER.append(one_Er(f_m=f_n, f_n=signal_sn))
    #     f_m_1 = signal_db8
    #     f_m_2 = signal_sym8
    # fig2, ax2 = plt.subplots()
    # ax2.plot(RSNR_db8, label='db8', marker='o', markersize=5)
    # ax2.plot(RSNR_sym8, label='sym8', marker='d', markersize=5)
    # ax2.legend(loc=1)
    # ax2.set_xlabel('Number of noise reduction iterations')
    # ax2.set_ylabel('Index value')
    # ax2.set_xticks([i for i in range(0, 19)])
    # ax2.set_xticklabels([i for i in range(1, 20)])
    # fig2.show()

#######################################
    # RSNR_20 = []
    # RSNR_50 = []
    # RSNR_120 = []
    # signal_20 = figure_data_a1_chayang + 0.5 * noise * 0.2
    # signal_50 = figure_data_a1_chayang + 0.5 * noise * 0.5
    # signal_120 = figure_data_a1_chayang + 0.5 * noise * 1.2
    # f_m_1 = signal_20
    # f_m_2 = signal_50
    # f_m_3 = signal_120
    #
    # for i in range(1, 20): #14不错 #20
    #     signall_20 = Waveleting(signal=signal_20)(i)
    #     signall_50 = Waveleting(signal=signal_50)(i)
    #     signall_120 = Waveleting(signal=signal_120)(i)
    #     f_n_1 = signall_20
    #     f_n_2 = signall_50
    #     f_n_3 = signall_120
    #     RSNR_20.append(1000*Rsnr(f_m=f_m_1, f_n=f_n_1))
    #     RSNR_50.append(1000*Rsnr(f_m=f_m_2, f_n=f_n_2))
    #     RSNR_120.append(1000*Rsnr(f_m=f_m_3, f_n=f_n_3))
    #     f_m_1 = signall_20
    #     f_m_2 = signall_50
    #     f_m_3 = signall_120
    # fig2, ax2 = plt.subplots()
    # ax2.plot(RSNR_20, label='20%', marker='o', markersize=5)
    # ax2.plot(RSNR_50, label='50%', marker='d', markersize=5)
    # ax2.plot(RSNR_120, label='120%', marker='s', markersize=5)
    # ax2.legend(loc=1)
    # ax2.set_xlabel('Number of noise reduction iterations')
    # ax2.set_ylabel('Index value')
    # ax2.set_xticks([i for i in range(0, 19)])
    # ax2.set_xticklabels([i for i in range(1, 20)])
    # fig2.show()

#######################gamma能谱#########################
    # gamma能谱
    figure_a1 = pd.read_excel(io='/Users/songyunlong/Desktop/7-a.xlsx', sheet_name='7-a', index_col=None, header=1)
    figure_data_a1 = figure_a1.values.T

    # print(figure_data_a1.shape)
    x_a1, y_a1 = np.split(ary=figure_data_a1, indices_or_sections=2, axis=0)
    # 插值
    spline = lambda x, y: interp1d(x, y, kind='linear')  # quadratic
    x_spline_new = np.linspace(np.min(x_a1), np.max(x_a1), 3000)
    # print(x_a1.shape, y_a1.shape)
    figure_data_a1_chayang = spline(x_a1.ravel(), y_a1.ravel())(x_spline_new)
    # x_a2, y_a2 = np.split(ary=figure_data_a2, indices_or_sections=2, axis=0)
    rng = np.random.RandomState(0)
    noise = rng.normal(loc=0, scale=1, size=figure_data_a1_chayang.shape)
    y_a1_addnoise = figure_data_a1_chayang + 0.05 * noise
    # 原始信号绘图
    # fig, ax = plt.subplots(nrows=2)
    # ax[0].plot(x_spline_new, figure_data_a1_chayang.ravel(), label='raw signal')
    # # ax[1].plot(x_a2.ravel(), y_a2.ravel())
    # ax[1].plot(x_spline_new, y_a1_addnoise.ravel(), label='add noise')
    # ax[0].legend()
    # ax[1].legend()
    # ax[0].set_xlabel('Number of noise reduction iterations')
    # ax[0].set_ylabel('Index value')
    # ax[1].set_xlabel('Number of noise reduction iterations')
    # ax[1].set_ylabel('Index value')
    # fig.show()

    # 小波迭代降噪
    # fig1, ax1 = plt.subplots()
    # signal = Waveleting(signal=y_a1_addnoise.ravel())(6)
    # # print(signal.shape)
    # ax1.plot(x_spline_new, signal, label='denoise 6th')
    # ax1.legend()
    # ax1.set_xlabel('Number of tracks')
    # ax1.set_ylabel('Counts')
    # fig1.show()