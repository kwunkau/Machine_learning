import pandas as pd
import numpy as np
dataset = pd.read_csv('watermelon_3.csv', delimiter=",")

# 处理数据集
attributeMap = {}
attributeMap['浅白'] = 0
attributeMap['青绿'] = 0.5
attributeMap['乌黑'] = 1
attributeMap['蜷缩'] = 0
attributeMap['稍蜷'] = 0.5
attributeMap['硬挺'] = 1
attributeMap['沉闷'] = 0
attributeMap['浊响'] = 0.5
attributeMap['清脆'] = 1
attributeMap['模糊'] = 0
attributeMap['稍糊'] = 0.5
attributeMap['清晰'] = 1
attributeMap['凹陷'] = 0
attributeMap['稍凹'] = 0.5
attributeMap['平坦'] = 1
attributeMap['硬滑'] = 0
attributeMap['软粘'] = 1
attributeMap['否'] = 0
attributeMap['是'] = 1
del dataset['编号']
dataset = np.array(dataset)
m, n = np.shape(dataset)
for i in range(m):
    for j in range(n):
        if dataset[i, j] in attributeMap:
            dataset[i, j] = attributeMap[dataset[i, j]]
        dataset[i, j] = round(dataset[i, j], 3)

trueY = dataset[:, n-1]
X = dataset[:, :n-1]
m, n = np.shape(X)

# P101,初始化参数
import random
d = n   # 输入向量的维数
l = 1   # 输出向量的维数
q = d+1   # 隐层节点的数量
theta = [random.random() for i in range(l)]   # 输出神经元的阈值
gamma = [random.random() for i in range(q)]   # 隐层神经元的阈值
# v size= d*q .输入和隐层神经元之间的连接权重
v = [[random.random() for i in range(q)] for j in range(d)]
# w size= q*l .隐藏和输出神经元之间的连接权重
w = [[random.random() for i in range(l)] for j in range(q)]
eta = 0.2    # 训练速度
maxIter = 5000  # 最大训练次数

import math
def sigmoid(iX,dimension):  # iX is a matrix with a dimension
    if dimension == 1:
        for i in range(len(iX)):
            iX[i] = 1 / (1 + math.exp(-iX[i]))
    else:
        for i in range(len(iX)):
            iX[i] = sigmoid(iX[i], dimension-1)
    return iX


# 标准BP
while(maxIter > 0):
    maxIter -= 1
    sumE = 0
    for i in range(m):
        alpha = np.dot(X[i], v)  # p101 line 2 from bottom, shape=1*q
        b = sigmoid(alpha-gamma, 1)  # b=f(alpha-gamma), shape=1*q
        beta = np.dot(b, w)  # shape=(1*q)*(q*l)=1*l
        predictY = sigmoid(beta-theta, 1)   # shape=1*l ,p102--5.3
        E = sum((predictY-trueY[i])*(predictY-trueY[i]))/2    # 5.4
        sumE += E  # 5.16
        # p104
        g = predictY*(1-predictY)*(trueY[i]-predictY)  # shape=1*l p103--5.10
        e = b*(1-b)*((np.dot(w, g.T)).T)  # shape=1*q , p104--5.15
        w += eta*np.dot(b.reshape((q, 1)), g.reshape((1, l)))  # 5.11
        theta -= eta*g  # 5.12
        v += eta*np.dot(X[i].reshape((d, 1)), e.reshape((1, q)))  # 5.13
        gamma -= eta*e  # 5.14
    # print(sumE)

# #累积 BP
# trueY=trueY.reshape((m,l))
# while(maxIter>0):
#     maxIter-=1
#     sumE=0
#     alpha = np.dot(X, v)#p101 line 2 from bottom, shape=m*q
#     b = sigmoid(alpha - gamma,2)  # b=f(alpha-gamma), shape=m*q
#     beta = np.dot(b, w)  # shape=(m*q)*(q*l)=m*l
#     predictY = sigmoid(beta - theta,2)  # shape=m*l ,p102--5.3
#
#     E = sum(sum((predictY - trueY) * (predictY - trueY))) / 2  # 5.4
#     # print(round(E,5))
#     g = predictY * (1 - predictY) * (trueY - predictY)  # shape=m*l p103--5.10
#     e = b * (1 - b) * ((np.dot(w, g.T)).T)  # shape=m*q , p104--5.15
#     w += eta * np.dot(b.T, g)  # 5.11 shape (q*l)=(q*m) * (m*l)
#     theta -= eta * g  # 5.12
#     v += eta * np.dot(X.T, e)  # 5.13 (d,q)=(d,m)*(m,q)
#     gamma -= eta * e  # 5.14


def predict(iX):
    alpha = np.dot(iX, v)  # p101 line 2 from bottom, shape=m*q
    b = sigmoid(alpha-gamma, 2)  # b=f(alpha-gamma), shape=m*q
    beta = np.dot(b, w)  # shape=(m*q)*(q*l)=m*l
    predictY = sigmoid(beta - theta, 2)  # shape=m*l ,p102--5.3
    return predictY

print(predict(X))