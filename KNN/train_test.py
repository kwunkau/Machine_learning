from sklearn import datasets
from train_test_split import train_test_split

# iris = datasets.load_iris()
# ## [[5.1 3.5 1.4 0.2][4.9 3.  1.4 0.2]....]
# X = iris.data
# y = iris.target
'''
## 方法1
## 使用concatenate函数进行拼接，因为传入的矩阵必须具有相同的形状。
## 因此需要对label进行reshape操作，reshape(-1,1)表示行数自动计算，1列。
## axis=1表示纵向拼接。
tempConcat = np.concatenate((X, y.reshape(-1,1)), axis=1)
## 拼接好后，直接进行乱序操作np.random.shuffle(tempConcat)
## 再将shuffle后的数组使用split方法拆分
shuffle_X,shuffle_y = np.split(tempConcat, [4], axis=1)
### 设置划分的比例
test_ratio = 0.2
test_size = int(len(X) * test_ratio)
X_train = shuffle_X[test_size:]
y_train = shuffle_y[test_size:]
X_test = shuffle_X[:test_size]
y_test = shuffle_y[:test_size]
'''

## 方法2封装数组随机分配
# X_train, X_test, y_train, y_test = train_test_split(X, y)
#
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)



# 手写数字数据集，封装好的对象，可以理解为一个字段
digits = datasets.load_digits()
# 可以使用keys()方法来看一下数据集的详情digits.keys()
print(digits.keys())

## 数据探索
# 特征的shape
X = digits.data
print(X.shape)  #(1797, 64)
# 标签的shape
y = digits.target
print(y.shape)  #(1797, )

# 标签分类digits.target_names
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])# 去除某一个具体的数据，查看其特征以及标签信息some_digit = X[666]
# some_digit
# array([ 0.,  0.,  5., 15., 14.,  3.,  0.,  0.,  0.,  0., 13., 15.,  9.,15.,  2.,  0.,  0.,  4., 16., 12.,  0., 10.,  6.,  0.,  0.,  8.,16.,  9.,  0.,  8., 10.,  0.,  0.,  7., 15.,  5.,  0., 12., 11.,0.,  0.,  7., 13.,  0.,  5., 16.,  6.,  0.,  0.,  0., 16., 12.,15., 13.,  1.,  0.,  0.,  0.,  6., 16., 12.,  2.,  0.,  0.])
# y[666]0# 也可以这条数据进行可视化some_digmit_image = some_digit.reshape(8, 8)
# plt.imshow(some_digmit_image, cmap = matplotlib.cm.binary)
# plt.show()