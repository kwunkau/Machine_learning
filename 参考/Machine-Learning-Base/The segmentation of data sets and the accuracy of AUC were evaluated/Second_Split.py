import random
import pandas as pd
import numpy as np

# 导入moice 100的数据集
header = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv('./ml-100k/u.data', sep='\t', names=header)
df = data.values.tolist()  # 将读取的CSV文件转化为二维矩阵

# 随机分配测试集和训练集
users = data.user_id.unique().shape[0]
items = data.item_id.unique().shape[0]
print('Number of users = ' + str(users) + ' | Number of movies = ' + str(items))
df = np.zeros((users, items))



# 留出法顺序分割 方法一:
def HoldOut(df, M):
    test = df[:M]
    train = df[M:]
    return train , test

# 留出法 方法二:
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2)

# K次交叉验证
from sklearn.model_selection import KFold
kf = KFold(n_splits=2)  # 设置k的次数
for train_index, test_index in kf.split(df):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_data_kf, test_data_kf = df[train_index], df[test_index]








# from sklearn.model_selection import KFold
# import numpy as np
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([1, 1, 1, 0, 5, 6])
# kf = KFold(n_splits=2)
#
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     print(X_train, X_test)
#     print('-------')
#     print( y_train, y_test)

#  自助法
def SplitData(df, M, k, seed):
    test = []
    train = []
    random.seed(seed)
    for users, items in df:
        if random.randint(0, M) == k:
            test.append([users, items])
        else:
            train.append([users, items])
    return train , test




