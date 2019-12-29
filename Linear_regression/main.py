from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor


# 加载波斯顿房价数据集
boston = datasets.load_boston()
X = boston.data
y= boston.target

# 排除异常数据
X = X[y < 50.0]
y = y[y < 50.0]

# 拆分训练数据集和测试数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 数据归一化
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)


# 实例化 SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5)

# 对训练数据集进行拟合
sgd_reg.fit(X_train_standard, y_train)

# 对测试数据集进行评分
print('模型评分：', sgd_reg.score(X_test_standard, y_test))