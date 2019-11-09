# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter

# raw_data_x是特征，raw_data_y是标签，0为良性，1为恶性
raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343853454, 3.368312451],
              [3.582294121, 4.679917921],
              [2.280362211, 2.866990212],
              [7.423436752, 4.685324231],
              [5.745231231, 3.532131321],
              [9.172112222, 2.511113104],
              [7.927841231, 3.421455345],
              [7.939831414, 0.791631213]
             ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
# 设置训练组
X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

# 将数据可视化
# X_train[y_train==0,0] 取y_train==0,0对于的第一个值
plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1], color='g', label='d1 a')
plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1], color='r', label='d2 b')
plt.xlabel('Tumor Size')
plt.ylabel('Time')
plt.axis([0,10,0,5])
#恶性肿瘤（绿色）、良性肿瘤（红色）
plt.show()

#判断x是良性肿瘤还是恶性肿瘤
x=[8.90933607318, 3.365731514]
distances = []  # 用来记录x到样本数据集中每个点的距离for x_train in X_train:

# 使用列表生成器，一行就能搞定，对于X_train中的每一个元素x_train都进行前面的运算，把结果生成一个列表
d = [sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train]
distances.append(d)
# print(distances)

# 排序,选k值
nearest = np.argsort(distances)
k = 6
topK_y = [y_train[i] for i in nearest[:k]]
print(topK_y)
#
# votes = Counter(topK_y)
# print(votes)