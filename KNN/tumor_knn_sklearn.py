from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score

# 读取数据 X, y
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

# 把数据分成训练数据和测试数据
raw_test_x = [[8.90933607318,3.365731514]
              ]

raw_test_y = [0]
X_test = np.array(raw_test_x)
y_test = np.array(raw_test_y)

# 构建KNN模型， K值为3、 并做训练
clf = KNeighborsClassifier(n_neighbors=6)
clf.fit(X_train, y_train)
print("预测值：%.2f" % (np.count_nonzero(clf.predict(X_test))))

# 计算准确率
correct = np.count_nonzero((clf.predict(X_test) == y_test) == True)
accuracy_score(y_test, clf.predict(X_test))
print("准确率(Accuracy) is: %.3f" % (correct / len(X_test)))
