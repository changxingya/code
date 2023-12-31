#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 21:14:41 2019

@author: xmos
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

fig, ax = plt.subplots()

lw = 0.5
""" Group 1 """
fpr_23 = np.array([0, 0, 0.01668, 0.01865, 0.02061, 0.03042, 0.04907, 0.08047, 0.12071, 0.158, 0.19823, 0.27772, 0.35721, 0.4524, 0.5054, 0.60059, 0.68989, 0.71541, 0.7949, 1])

""" Group 2 """
fpr_45 = np.array([0, 0.00704, 0.01207, 0.06036, 0.08753, 0.11066, 0.13783, 0.14789, 0.18511, 0.25252, 0.26157, 0.31087, 0.40744, 0.41147, 0.45674, 0.52314, 0.58048, 0.5996, 0.64588, 0.67907, 0.69618, 0.82797, 0.91147, 1])

# """ Sparse reconstruction """
# tpr_4 = np.array([0, 0.66162, 0.6832, 0.79422, 0.84298, 0.87765, 0.89401, 0.89807, 0.91184, 0.93337, 0.93591, 0.94766, 0.95754, 0.9577, 0.96006, 0.96859, 0.97934, 0.98301, 0.98883, 0.99093, 0.99174, 0.99594, 0.99798, 1])
#
# ax.plot(fpr_45, tpr_4, 'o-', lw=lw, color='black', markerfacecolor='teal', label='Sparse reconstruction')


""" AMDN """
fpr_1 = np.array([0, 0, 0.01668, 0.01865, 0.02061, 0.03042, 0.04907, 0.05103, 0.08047, 0.12071, 0.158, 0.19823, 0.2002, 0.2051, 0.27772, 0.35721, 0.4524, 0.46124, 0.5054, 0.60059, 0.60255, 0.68989, 0.71541, 0.7949, 1])
tpr_1 = np.array([0, 0.01659, 0.37876, 0.4221, 0.46409, 0.58149, 0.6761, 0.6837, 0.75, 0.81077, 0.84392, 0.86991, 0.87117, 0.87431, 0.91575, 0.93923, 0.97099, 0.9746, 0.98757, 0.993, 0.99309, 0.99669, 0.99724, 0.99832, 1])

ax.plot(fpr_1, tpr_1, 'v-', lw=lw, color='navy', markerfacecolor='mediumorchid', label='AMDN')

""" SCLF """
fpr_8 = np.array([0, 0.0019, 0.00475, 0.02169, 0.03968, 0.05571, 0.07678, 0.1049, 0.12999, 0.15408, 0.19732, 0.27377, 0.38643, 0.5373, 0.69824, 0.82497, 1])
tpr_8 = np.array([0, 0.1608 , 0.27961, 0.44077, 0.5551 , 0.62397, 0.6832 , 0.7314 , 0.79477, 0.84711, 0.86088, 0.87052, 0.88981, 0.92011, 0.95041, 0.97934, 1])

ax.plot(fpr_8, tpr_8, '*-', lw=lw, color='maroon', markerfacecolor='teal', label='SCLF')


""" MDT """
fpr_6 = np.array([0, 0.01478, 0.02675, 0.04378, 0.06286, 0.08896, 0.11608, 0.16933, 0.22461, 0.28895, 0.36033, 0.44078, 0.55441, 0.65901, 0.74852, 1])
tpr_6 = np.array([0, 0.30992, 0.40771, 0.47796, 0.51377, 0.57851, 0.62121, 0.68182, 0.73278, 0.78237, 0.82782, 0.86088, 0.9146, 0.94077, 0.9697, 1])

ax.plot(fpr_6, tpr_6, 'H-', lw=lw, color='saddlebrown', markerfacecolor='orange', label='MDT')

""" SFMPPCA """
tpr_3 = np.array([0, 0, 0.08702, 0.10537, 0.16298, 0.26902, 0.3384, 0.4066, 0.46925, 0.51691, 0.56215, 0.64123, 0.71404, 0.79873, 0.84629, 0.92299, 0.96823, 0.97601, 0.99085, 1])

ax.plot(fpr_23, tpr_3, 'X-', lw=lw, color='saddlebrown', markerfacecolor='gold', label=r'Social Force + MPPCA')

""" SF """
fpr_7 = np.array([0, 0.00896, 0.02798, 0.05506, 0.10222, 0.1916, 0.33125, 0.53435, 0.73542, 1	])
tpr_7 = np.array([0, 0.09229, 0.18457, 0.2741, 0.38843, 0.54545, 0.73278, 0.86088, 0.99587, 1])

ax.plot(fpr_7, tpr_7, 'd-', lw=lw, color='darkslategrey', markerfacecolor='steelblue', label='Social Force')

""" MPPCA """
tpr_2 = np.array([0, 0, 0.07276, 0.07597, 0.07892, 0.0916, 0.11292, 0.15884, 0.25466, 0.3186, 0.37714, 0.48013, 0.5762, 0.68625, 0.74557, 0.84807, 0.9338, 0.95326, 0.98343, 1	])

ax.plot(fpr_23, tpr_2, 's-', lw=lw, color='black', markerfacecolor='cornflowerblue', label='MPPCA')


""" Adam """
tpr_5 = np.array([0, 0.01265, 0.02172, 0.1157, 0.1796, 0.22452, 0.26353, 0.27688, 0.33058, 0.45705, 0.47107, 0.53795, 0.64994, 0.65427, 0.70098, 0.75895, 0.79314, 0.80292, 0.82645, 0.84351, 0.85233, 0.92041, 0.96281, 1])

ax.plot(fpr_45, tpr_5, '>-', lw=lw, color='black', markerfacecolor='gray', label='Adam')

""" Our """
fpr_9 = np.array([0,0.012882,0.017034,0.025754,0.052591,0.081328,0.122976,0.161503,0.21289,0.249856,0.307769,0.365697,0.433286,0.504104,0.592641,0.655428,0.748817,0.908228,1])
tpr_9 = np.array([0,5.7e-05,0.211064,0.50341,0.670561,0.74761,0.815925,0.853457,0.897639,0.91978,0.937617,0.951058,0.964542,0.975842,0.985023,0.989694,0.992302,0.995201,1])
ax.plot(fpr_9, tpr_9, '*-', lw=lw, color='maroon', markerfacecolor='red', label='Our')




ax.plot([0, 1], [1, 0], color='grey', lw=0.5, linestyle='--')
ax.set_xlim([-0.01, 1.0])
ax.set_ylim([0.0, 1.01])
ax.set_xlabel('False Positive Rate',fontsize=17)
ax.set_ylabel('True Positive Rate',fontsize=17)
ax.legend(loc="lower right")
ax.grid(color='grey', linestyle='-.',lw=0.5)
#plt.axis('equal')
plt.savefig('./ped1_fl_curves.png', dpi=1000, bbox_inches='tight')
plt.show()