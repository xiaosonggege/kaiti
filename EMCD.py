#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: EMCD
@time: 2021/6/14 11:57 上午
'''
import numpy as np
import sys
sys.path.append('/root/PyEMD/')
from PyEMD import EMD, Visualisation

class EMDblock:
    def __init__(self, signal:np.ndarray, imfs_num:int=-1):
        self._signal = signal
        self._imfs_num = imfs_num

    def __call__(self, *args, **kwargs):
        return self.fenjie()

    @property
    def signal(self):
        return self._signal
    @signal.setter
    def signal(self, signal:np.ndarray):
        self._signal = signal

    @property
    def imfs_num(self):
        return self._imfs_num
    @imfs_num.setter
    def imfs_num(self, imfs_num:int):
        self._imfs_num = imfs_num

    def fenjie(self):
        emd = EMD()
        emd.emd(self._signal, max_imf=self._imfs_num)
        imfs, res = emd.get_imfs_and_residue()
        return imfs, res

    def plot(self, imfs:np.ndarray, res:np.ndarray):
        t = np.arange(self._signal.shape[0])
        vis = Visualisation()
        vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
        vis.plot_instant_freq(t, imfs=imfs)
        vis.show()



if __name__ == '__main__':
    # t = np.arange(0, 3, 0.01)
    # S = np.sin(13 * t + 0.2 * t ** 1.4) - np.cos(3 * t)
    # # Extract imfs and residue
    # # In case of EMD
    # emd = EMD()
    # emd.emd(S, max_imf=2)
    # imfs, res = emd.get_imfs_and_residue()
    # print(imfs.data, imfs.shape, imfs.__class__)
    # print(res.data, res.shape)
    # # In general:
    # # components = EEMD()(S)
    # # imfs, res = components[:-1], components[-1]
    # vis = Visualisation()
    # vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
    # vis.plot_instant_freq(t, imfs=imfs)
    # vis.show()
    pass
