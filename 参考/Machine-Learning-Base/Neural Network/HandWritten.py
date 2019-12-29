import numpy as np
from sklearn.datasets import load_digits  # 数字数据集
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer  # 标签二值化
from NeuralNetwork import NeuralNetwork
from sklearn.cross_validation import train_test_split


digits = load_digits()  # 导入数据，sklearn自带的
X = digits.data  # 表示特征量
y = digits.target  # 表示标签
X -= X.min()  # 归一化，范围0-1
X /= X.max()

# 输入层：每个实例都有64个像素点（即特征值），输出层：都有10个标签。隐藏层通常要比输入层多一些。
nn = NeuralNetwork([64, 100, 10], 'logistic')
X_train, X_test, y_train, y_test = train_test_split(X, y)  # 划分训练集和测试集
labels_train = LabelBinarizer().fit_transform(y_train)  # 标签二值化
labels_test = LabelBinarizer().fit_transform(y_test)
print("start fitting")
nn.fit(X_train, labels_train, epochs=3000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))   # 分类概率最大

print(confusion_matrix(y_test,  predictions))
print(classification_report(y_test, predictions))