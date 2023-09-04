#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:03:03 2019

@author: xmos
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

fig, ax = plt.subplots(figsize=(7,6))

lw = 1

""" Sparse reconstruction """
fpr_0 = np.array([0, 0.02203, 0.05812, 0.08317, 0.1042, 0.12822, 0.15524, 0.19229, 0.20527, 0.23831, 0.26933, 0.30332, 0.31933, 1])
tpr_0 = np.array([0, 0.03804, 0.05707, 0.07745, 0.11005, 0.1644, 0.2269, 0.2894, 0.34239, 0.40761, 0.48098, 0.62256, 0.63179, 1])

ax.plot(fpr_0, tpr_0, 'o-', lw=lw, color='black', markerfacecolor='teal', label='Sparse reconstruction')

""" AMDN """
fpr_1 = np.array([0, 0.02863, 0.04146, 0.05627, 0.09378, 0.15005, 0.20336, 0.3613, 0.61895, 0.70681, 0.80652, 0.89141, 1])
tpr_1 = np.array([0, 0.28318, 0.3442, 0.37392, 0.40991, 0.45372, 0.48501, 0.5648, 0.69622, 0.75411, 0.84485, 0.91369, 1])

ax.plot(fpr_1, tpr_1, 'v-', lw=lw, color='navy', markerfacecolor='mediumorchid', label='AMDN')

""" SCLF """
fpr_2 = np.array([0, 0.01974, 0.06022, 0.09773, 0.15005, 0.20138, 0.2922, 0.38401, 0.48272, 0.5617, 0.64659, 0.72162, 0.77887, 0.86081, 0.92991, 1])
tpr_2 = np.array([0, 0.23781, 0.27379, 0.2957, 0.35046, 0.38801, 0.43181, 0.5116, 0.58201, 0.63051, 0.70404, 0.85111, 0.90117, 0.94342, 0.96063, 1])

ax.plot(fpr_2, tpr_2, '*-', lw=lw, color='maroon', markerfacecolor='red', label='SCLF')

""" MDT """
fpr_3 = np.array([0, 0.03159, 0.04936, 0.06022, 0.07206, 0.10168, 0.13327, 0.17769, 0.30109, 0.42152, 0.58243, 0.89832, 1])
tpr_3 = np.array([0, 0.01721, 0.03598, 0.07353, 0.11578, 0.18149, 0.22842, 0.28631, 0.35671, 0.40052, 0.45372, 0.67744, 1])

ax.plot(fpr_3, tpr_3, 'H-', lw=lw, color='saddlebrown', markerfacecolor='orange', label='MDT')

""" SFMPPCA """
fpr_4 = np.array([0, 0.15597, 0.17473, 0.18855, 0.20237, 0.23593, 0.27838, 0.33366, 0.40276, 0.51629, 0.61007, 0.68806, 0.78875, 0.85489, 0.9309, 0.97927, 0.98914, 0.99506, 1])
tpr_4 = np.array([0, 0.10795, 0.11963, 0.12516, 0.13142, 0.14081, 0.15489, 0.16741, 0.17992, 0.21434, 0.24563, 0.27379, 0.291, 0.30352, 0.32699, 0.36767, 0.40052, 0.44433, 1])

ax.plot(fpr_4, tpr_4, 'X-', lw=lw, color='saddlebrown', markerfacecolor='gold', label=r'Social Force + MPPCA')

""" SF """
fpr_5 = np.array([0, 0.10958, 0.13031, 0.17868, 0.22014, 0.27937, 0.38203, 0.50049, 0.66338, 0.76999, 0.8618, 1])
tpr_5 = np.array([0, 0.07666, 0.08605, 0.097, 0.10795, 0.12673, 0.14237, 0.17679, 0.18618, 0.20339, 0.22842, 1])

ax.plot(fpr_5, tpr_5, 'd-', lw=lw, color='darkslategrey', markerfacecolor='steelblue', label='Social Force')

""" MPPCA """
fpr_6 = np.array([0, 0.66239, 0.71076, 0.73149, 0.75321, 0.78381, 0.81343, 0.83317, 0.86575, 0.89042, 0.91313, 0.93287, 1])
tpr_6 = np.array([0, 0.14863, 0.16035, 0.16741, 0.17366, 0.17992, 0.18462, 0.18774, 0.194, 0.194, 0.20026, 0.20339, 1])

ax.plot(fpr_6, tpr_6, 's-', lw=lw, color='black', markerfacecolor='cornflowerblue', label='MPPCA')

""" Adam """
fpr_7 = np.array([0, 0.10832, 0.17251, 0.22867, 0.27982, 0.33397, 0.40719, 0.45934, 0.49544, 0.54157, 0.58269, 0.61678, 0.64987, 0.69198, 0.72107, 1])
tpr_7 = np.array([0, 0.00951, 0.01223, 0.01766, 0.0231, 0.02717, 0.03533, 0.03804, 0.04484, 0.05435, 0.06114, 0.07201, 0.08016, 0.08526, 0.09783, 1])

ax.plot(fpr_7, tpr_7, '>-', lw=lw, color='black', markerfacecolor='gray', label='Adam')

""" Ours """
fpr_8 = np.array([0, 0.01102, 0.02203, 0.03039, 0.04147, 0.05256, 0.06085, 0.07191, 0.10459,0.14532, 0.18338,0.22952,0.29721,0.36484,0.42972,0.82702,1])
tpr_8 = np.array([0, 0.0788, 0.14946, 0.24457, 0.34511, 0.44565, 0.5163, 0.60598, 0.69837, 0.76902, 0.85054, 0.9212, 0.96739,0.99185, 0.99728,0.99728,1])

ax.plot(fpr_8, tpr_8, '', lw=lw, color='', markerfacecolor='', label='Ours')


ax.plot([0, 1], [1, 0], color='grey', lw=0.5, linestyle='--')
ax.set_xlim([-0.01, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="upper left")
ax.grid(color='grey', linestyle='-.',lw=0.5)

#fig.savefig('/Users/xmos/Documents/NutSync/MyOwn/AnomalyDetection/Pics/ped1_pl_curves_.png', dpi=300, bbox_inches='tight')
plt.show()