import numpy as np

def fitSLR(x, y):
    """
    训练简单线性模型
    """
    n = len(x)       # 获取数据集长度
    dinominator = 0  # 分母
    numerator = 0    # 分子
    for i in range(0, n):
        numerator += (x[i] - np.mean(x))*y[i]  # mean均值
        dinominator += (x[i] - np.mean(x))**2
    b1 = numerator/float(dinominator)  # 回归线斜率
    b0 = np.mean(y)-b1*np.mean(x)      # 回归线截距
    return b0, b1

def predict(x, b0, b1):
    """
    根据学习算法做预测
    """
    return b0 + x*b1

x = [1, 3, 2, 1, 3]
y = [14, 24, 18, 17, 27]

b0, b1 = fitSLR(x, y)

print("intercept:", b0, " slope:", b1)

x_test = 6

y_test = predict(6, b0, b1)

print("y_test:", y_test)

# 画图
import matplotlib.pyplot as plt

y_perd = b0 + b1*np.array(x)

plt.scatter(x, y)  # 散点图
plt.plot(x, y_perd, color='black')
plt.xlabel("x")
plt.ylabel("y")
plt.axis([0, 6, 0, 30])  # 设置横纵坐标的范围
plt.show()
