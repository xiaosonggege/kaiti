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

def erase_basic_line(signal:np.ndarray):
    """
    :param signal:
    :return:
    """
    #翻转信号
    signal_r = -signal
    peaks, _ = find_peaks(signal_r)
    #基线
    bl = np.min(peaks)
    for i in range(peaks.__len__()-1):
        min_point = np.minimum(peaks[i], peaks[i+1])
        signal[i:i+1+1] += min_point - bl
    return signal, peaks

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
    signal = np.array([])
    signal_erase_basic_line, peaks = erase_basic_line(copy.deepcopy(signal))
    area_array = peaks_area(signal_erase_basic_line, peaks)
