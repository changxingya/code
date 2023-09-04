import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np






a=[1,2,3]
b = [4,5,6]
c=[7, 8, 9]
d = [3, 6, 8]


f = open('data.npy', 'wb')
for j in range(4):
    if j == 0:
        temp = a
    elif j == 1:
        temp = b
    elif j ==2:
        temp = c
    else:
        temp = d
    np.save(f, temp)

x = np.load('../Txt/data.npz')
print(x)
