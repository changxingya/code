#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 09:15:09 2019

@author: xmos
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

fig, ax = plt.subplots(figsize=(7,6))

fsize = 12

methods = ['Adam', 'SF','SF+MPPCA','MPPCA','MDT','SCLF','AMDN','Sparse reconstruction']

p1_re = np.array([0.38, 0.31, 0.32, 0.4, 0.25, 0.15, 0.16, 0.12])

p2_re = np.array([0.42, 0.42, 0.36, 0.3, 0.25, 0.01, 0.15, 0.01])

colors = ['gray','steelblue','gold','cornflowerblue','orange','darkred','mediumorchid','teal']

y = np.arange(22).reshape((2,-1)).T
y = y[-1::-1, -1::-1]

for i in range(9):
    ax.barh(y[i+1], np.array([p1_re[i],p2_re[i]]), height=0.9, color=colors[i], label=methods[i])
    
for a,b in zip(p1_re, y[1:-1,0]):
    if a == 0.01:
        ax.text(a+0.01, b, '- -', va='center', fontsize=fsize)
    else:
        ax.text(a+0.01, b, '%.2f' % a, va='center', fontsize=fsize)
    
for a,b in zip(p2_re, y[1:-1,1]):
    if a == 0.01:
        ax.text(a+0.01, b, '- -', va='center', fontsize=fsize)
    else:
        ax.text(a+0.01, b, '%.2f' % a, va='center', fontsize=fsize)

ax.set_yticks([16, 5])    
ax.set_yticklabels(['Ped1', 'Ped2'])
ax.set_xlim([0.0, 0.8])
ax.legend(loc="upper right")
ax.grid(axis='x', color='grey', linestyle='-.',lw=0.5)
ax.set_xlabel('Equal Error Rate')
plt.show()
