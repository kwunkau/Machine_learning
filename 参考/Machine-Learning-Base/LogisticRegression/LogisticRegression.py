import numpy as np
import random

# 梯度下降算法函数,x/y是输入变量，theta是参数，alpha是学习率，m是实例，numIterations梯度下降迭代次数
def gradientDescent(x, y, theta, alpha, m, numIterations):
    """
    梯度下降算法拟合逻辑回归参数值
    numpy乘法运算中"*"是数组元素逐个计算
    dot是按照矩阵乘法的规则来运算
    """
    xTrans = x.transpose()  # 转置矩阵
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)  # 矩阵相乘
        loss = hypothesis - y  # 预测值-真实值
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        # 成本函数：loss方差的平均加总，在这里采用了常用的成本函数，而非logistic特有的
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m  #计算梯度
        # update
        theta = theta - alpha * gradient  #参数theta的计算，即更新法则
    return theta


def genData(numPoints, bias, variance):
    """
    创建数据集，numPoints：实例数，bias：偏差，variance：方差
    """
    x = np.zeros(shape=(numPoints, 2))  # 特征值，numPoints条样本数据，2个特征。numPoints行，2列。
    y = np.zeros(shape=numPoints)       # 标签(label)，numPoints行，1列。
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i][0] = 1  # 第i行第1列全部等于1
        x[i][1] = i  # 第i行第2列等于i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance  # 第i行第2列等于i+bias(偏差），再加,0-1的随机数，以及方差
    return x, y

x, y = genData(100, 25, 10)  #传入参数
m, n = np.shape(x)
n_y = np.shape(y)
print("x shape:", str(m), " ", str(n))
print("y length:", str(n_y))

numIterations= 100000  # 迭代次数
alpha = 0.0005         # alpha learning rate 学习率
theta = np.ones(n)     # theta 逻辑回归参数
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)
