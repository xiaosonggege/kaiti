#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: basic_line
@time: 2020/9/8 2:27 下午
'''
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, find_peaks_cwt
import copy
from matplotlib import pyplot as plt
from xiaobo import Waveleting, Rsnr
from scipy.interpolate import interp1d

def erase_basic_line(signal:np.ndarray, height:float, threshold:float=None, distance:float=None, prominence:float=None, width:float=None):
    """
    :param signal: 信号
    :param height: 低于指定height的信号都不考虑
    :param threshold: 其与相邻样本的垂直距离
    :param prominence: 相邻峰之间的最小水平距离，先移除较小的峰，直到所有剩余峰的条件都满足为止
    :param width: 波峰宽度
    :return:
    """
    #翻转信号
    signal_r = -signal
    peaks, _ = find_peaks(signal_r, height=height) #threshold=threshold, distance=distance, prominence=prominence, width=width
    return peaks

def peaks_area(signal:np.ndarray, peaks:np.ndarray):
    """

    :param signal:
    :param peaks:
    :return:
    """
    area_array = np.array([])
    for i in range(peaks.__len__()-1):
        area_array = np.array([np.sum(signal[peaks[i], peaks[i+1+1]])]) if area_array.size == 0 else \
            np.hstack((area_array, np.array([np.sum(signal[peaks[i], peaks[i+1+1]])])))
    return area_array

if __name__ == '__main__':
    # gamma能谱
    figure_a1 = pd.read_excel(io='/Users/songyunlong/Desktop/实验室/寿老师项目/能谱分析/伽马谱仿真结果.xlsx', sheet_name='Sheet1', index_col=None, header=None)
    # print(figure_a1)
    figure_data_a1 = figure_a1.values.T
    x_a1, y_a1 = np.split(ary=figure_data_a1, indices_or_sections=2, axis=0)
    x_a1 = x_a1.ravel()
    y_a1 = y_a1.ravel()
    # print(x_a1.shape, y_a1.shape)

    #小波降噪后的信号图
    signal = Waveleting(signal=y_a1, threshold=1e-3)(6)
    # print(np.max(signal))
    # fig, ax = plt.subplots(nrows=2)
    # ax[0].plot(x_a1, y_a1)
    # ax[1].plot(x_a1, signal)
    # fig.show()
    #
    peaks = erase_basic_line(copy.deepcopy(signal), height=-0.2*1e-7, width=7, distance=4)
    # print(peaks.shape)
    peaks = np.insert(peaks, 0, 0)
    peaks = np.append(peaks, x_a1.__len__()-1)
    # print(peaks.shape)
    # 插值
    spline = lambda x, y: interp1d(x, y, kind='quadratic')  # quadratic
    x_spline_new = np.linspace(np.min(x_a1[peaks]), np.max(x_a1[peaks]), signal.shape[0])

    #插样后的基线图线
    peaks_chayang = spline(x_a1[peaks], signal[peaks])(x_spline_new)

    #去基线
    qujixian_result = signal - peaks_chayang

    # 峰值点
    peaks_high, _ = peak_id, peak_property = find_peaks(qujixian_result, height=1e-9, prominence=1e-8)
    # print(peaks_high)
    # print(x_a1[peaks_high])
    ######绘制去基线后的图
    # fig, ax = plt.subplots()
    # ax.plot(x_a1, qujixian_result)
    # ax.scatter(x_a1[peaks_high], qujixian_result[peaks_high], color='r')
    # fig.show()

    ######找峰值区域的起点和终点
    #峰值及左右边界
    peaks_range = np.insert(peaks_high, 0, x_a1[0])
    peaks_range = np.append(peaks_range, x_a1.__len__()-1)
    #峰值区域边界点高度阈值
    Th = 1e-7 * 1e-3
    peaks_start = []
    # print(peaks_range)
    #向左遍历
    for i in range(1, peaks_range.__len__()-1):
        for j in reversed(range(peaks_range[i-1], peaks_range[i])):
            if qujixian_result[j] <= Th:
                peaks_start.append(j)
                break

    peaks_end = []
    #向右遍历
    for i in range(1, peaks_range.__len__()-1):
        for j in range(peaks_range[i], peaks_range[i+1]):
            if qujixian_result[j] <= Th:
                peaks_end.append(j)
                break

    peaks_start_end = np.hstack((np.array(peaks_start), np.array(peaks_end)))
    peaks_start_end = np.sort(peaks_start_end)
    # print(peaks_start_end.shape, peaks_high.shape)
    # print(peaks_start_end)

    ################计算峰值区域面积、本地面积和净峰面积###############
    # 峰值区域面积
    fengzhimianji = np.array([])

    # 本地面积
    bendimianji = np.array([])

    # 净峰面积
    jingfengmianji = np.array([])
    for i in range(0, peaks_start_end.shape[0], 2):
        # print('(%s, %s)' % (peaks_start_end[i], peaks_start_end[i+1]))
        area_fengzhimianji = np.sum(signal[peaks_start_end[i]:peaks_start_end[i+1]+1])
        area_bendimianji = (signal[peaks_start_end[i]] + signal[peaks_start_end[i+1]]) * (peaks_start_end[i+1] - peaks_start_end[i] + 1) / 2
        area_jingfengmianji = area_fengzhimianji - area_bendimianji
        jingfengmianji = np.append(jingfengmianji, area_jingfengmianji)

        bendimianji = np.append(bendimianji, area_bendimianji)
        # print(area_fengzhimianji)
        fengzhimianji = np.append(fengzhimianji, area_fengzhimianji)
    print(fengzhimianji)
    print(bendimianji)
    print(jingfengmianji)



    ######绘制去基线后的图以及峰值区域########
    # fig, ax = plt.subplots()
    # ax.plot(x_a1, qujixian_result)
    # ax.scatter(x_a1[peaks_high], qujixian_result[peaks_high], color='r')
    # ax.scatter(x_a1[peaks_start_end], qujixian_result[peaks_start_end], color='black', s=4)
    # fig.show()



    # area_array = peaks_area(signal_erase_basic_line, peaks)
    # print(x_a1[peaks])
    # print(signal[peaks].shape)

    #################基线和去噪后的信号绘制################
    # fig, ax = plt.subplots()
    # ax.plot(x_a1, signal)
    # ax.plot(x_a1, peaks_chayang, color='r')
    # fig.show()

    # 小波迭代降噪rsnr统计图像
    # RSNR = []
    # f_m = y_a1
    # for i in range(1, 16):  # 6不错 #16
    #     signal = Waveleting(signal=y_a1)(i)
    #     f_n = signal
    #     RSNR.append(Rsnr(f_m=f_m, f_n=f_n))
    #     f_m = signal
    # fig2, ax1 = plt.subplots()
    # ax1.plot(RSNR, label='RSNR', marker='o', markersize=5)
    # ax1.legend(loc=1)
    # ax1.set_xlabel('Number of noise reduction iterations')
    # ax1.set_ylabel('Index value')
    # ax1.set_xticks([i for i in range(0, 16)])
    # ax1.set_xticklabels([i for i in range(0, 16)])
    #
    # fig2.show()