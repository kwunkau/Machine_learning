import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)  # 输出结果显示全部列

# 读取数据，从第1行开始
data = pd.read_csv(r"ccpp.csv", header=0)
# 显示数据的前五行，如果是最后五行，用data.tail()
print(data.head())
# 查看描述性统计,只能看数值型数据.
print(data.describe())

# 显示数据的维度
print(data.shape)
# 现在我们开始准备样本特征X，我们用AT， V，AP和RH这4个列作为样本特征
x = data[['AT', 'V', 'AP', 'RH']]
# 接着我们准备样本输出y， 我们用PE作为样本输出
y = data[['PE']]

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# 查看训练集和测试集的维度
# print('x_train.shape:', x_train.shape)
# print('x__test.shape:', x_test.shape)
# print('y_train.shape:', y_train.shape)
# print('y_test.shape:', y_test.shape)

#  从sklearn库中导入线性回归函数
from sklearn.linear_model import LinearRegression
# 执行函数获得一个线性回归模型
LR = LinearRegression()  # 这是一个未经训练的机器学习模型
# 对模型传入输入数据x_train和输出数据y_train
LR.fit(x_train, y_train)  # 这是一个经过训练的机器学习模型

# 输出线性回归的截距和各个系数
print('LR.intercept_:', LR.intercept_)
print('LR.coef_:', LR.coef_)

# 评价模型。这里使用MSE和RMSE来评价模型的好坏
y_pred = LR.predict(x_test)
# 引入sklearn模型评价工具库
from sklearn import metrics
print("MSE: ", metrics.mean_squared_error(y_test, y_pred))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# 画散点图
ax.scatter(y_test, y_pred)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# 设置标题
ax.set_title('Plot')
# 设置X轴标签
ax.set_xlabel('Measured')
# 设置Y轴标签
ax.set_ylabel('Predicted')
# 显示所画的图
plt.show()