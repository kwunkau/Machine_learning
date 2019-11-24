import numpy as np
# 创建100个随机数
x = np.random.randint(0,100,size=100)

# 最值归一化（向量）
# 最值归一化公式，映射到0，1之间
(x - np.min(x)) / (np.max(x) -  np.min(x))

# 最值归一化（矩阵）
# 0～100范围内的50*2的矩阵
X = np.random.randint(0,100,(50,2))
# 将矩阵改为浮点型
X = np.array(X, dtype=float)
# 最值归一化公式，对于每一个维度（列方向）进行归一化。
# X[:,0]第一列，第一个特征
X[:,0] = (X[:,0] - np.min(X[:,0])) / (np.max(X[:,0]) - np.min(X[:,0]))
# X[:,1]第二列，第二个特征
X[:,1] = (X[:,1] - np.min(X[:,1])) / (np.max(X[:,1]) - np.min(X[:,1]))

# 如果有n个特征，可以写个循环：
for i in range(0,2):
    X[:,i] = (X[:,i]-np.min(X[:,i])) / (np.max(X[:,i] - np.min(X[:,i])))