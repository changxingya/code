import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from pylab import *


def roc(y_test, y_score, k):
    #roc,eer
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_score)#
    roc_auc = auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    print('auc:', roc_auc)
    print('eer:', eer)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f' % roc_auc+', eer = %0.2f' % eer)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    ax = plt.gca()
    ax.set_aspect(1)
    plt.savefig("../pre_label/roc_{}.png".format(k))
    return roc_auc, eer

def cal_original(img):
    count = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 255:
                count = count + 1
    return count

def Cal_ROI(input, generate):
    pixel_count = 0
    gr_pixel_num  = cal_original(input)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i][j] == 255 and generate[i][j] > 0:
                pixel_count = pixel_count + 1
    pro_score = pixel_count / (gr_pixel_num + 1e-12)
    if pro_score > 0.400:
        return 1
    else:
        return -1

def resize_image(img):
    img = cv2.medianBlur(img.astype(np.float32), 3)
    img = cv2.dilate(img.astype(np.float32), (5, 5))
    img = cv2.pyrUp(img)
    img = cv2.pyrUp(img)
    img = cv2.resize(img, (360, 240))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 0 :
                img == 1
    return img

def plot_line_chart(x_arry, y_arry):
    if len(x_arry) != len(y_arry):
        raise Exception("input array length is not equal to the output")
    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.plot(x_arry, y_arry, 'ro-', color='#4169E1', alpha=0.8, linewidth=1)

    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    # plt.legend(loc="upper right")
    # plt.xlabel('one-class SVM parameter u ')
    plt.xlabel('Number of feature tensors channel ')
    plt.ylabel('Frame-level AUC')
    plt.savefig("/Users/changxingya/Desktop/论文工作/第一篇/paper_img/mu.png")

    plt.show()


