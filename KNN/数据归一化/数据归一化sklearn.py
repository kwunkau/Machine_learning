from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 加载鸢尾花数据集
iris = datasets.load_iris()

# 提取数据集中的特征数据
X = iris.data
y = iris.target

# 把数据集划分为训练数据集和测试数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 对象实例化
stardardScaler = StandardScaler()

# 类似于模型的训练过程
stardardScaler.fit(X_train)

# 使用 transform 实现均值方差归一化
X_train_scale = stardardScaler.transform(X_train)

# 不要对X_test进行训练，直接调用前面训练好的模型进行归一化
X_test_scale = stardardScaler.transform(X_test)

# 调用 K-近邻算法
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scale, y_train)

# 对算法进行评分
print('算法评分:', knn.score(X_test_scale, y_test))