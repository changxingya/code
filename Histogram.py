# import numpy as np
# import matplotlib.pyplot as plt
# import random
#
# # 准备数据
# x_data = [5,10, 15, 20, 25]
# y_data = [96.7,97.5,98.6, 98.0, 97.6]
#
#
# # 正确显示中文和负号
# plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False
#
# # 画图，plt.bar()可以画柱状图
# for i in range(len(x_data)):
# 	plt.bar(x_data[i], y_data[i])
# # 设置图片名称
# plt.title("销量分析")
# # 设置x轴标签名
# plt.xlabel("年份")
# # 设置y轴标签名
# plt.ylabel("销量")
# # 显示
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

x = ()
y1 = []
y2 = []

bar_width = 0.6  # 条形宽度

plt.bar(x, y1, bar_width, color='dodgerblue', label='y1')
plt.bar(x, y2, bar_width,color= 'tomato', bottom=y1,  label='y2')# 堆叠在第一个上方

plt.legend()
plt.show()
