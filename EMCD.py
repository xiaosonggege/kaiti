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

class Name:
    pass


if __name__ == '__main__':
    t = np.arange(0, 3, 0.01)
    S = np.sin(13 * t + 0.2 * t ** 1.4) - np.cos(3 * t)

    # Extract imfs and residue
    # In case of EMD
    emd = EMD()
    emd.emd(S)
    imfs, res = emd.get_imfs_and_residue()

    # In general:
    # components = EEMD()(S)
    # imfs, res = components[:-1], components[-1]

    vis = Visualisation()
    vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
    vis.plot_instant_freq(t, imfs=imfs)
    vis.show()
