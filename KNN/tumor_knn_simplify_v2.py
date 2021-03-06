# -*- coding: utf-8 -*-
import numpy as np
import math as sqrt
from collections import Counter
from metrics import accuracy_score


class kNNClassifier:
    def __init__(self, k):
        """初始化分类器"""
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k"
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict结果的向量"""
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        """根据X_test进行预测, 给出预测的真值y_test，计算预测模型的准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "kNN(k=%d)" % self.k


if __name__ == '__main__':
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

    X_train = np.array(raw_data_X)
    y_train = np.array(raw_data_y)
    knn_clf = kNNClassifier(k=6)
    knn_clf.fit(X_train, y_train)
    # 判断x是良性肿瘤还是恶性肿瘤
    raw_test_x = [[8.90933607318, 3.365731514]]
    x = np.array(raw_test_x)
    X_predict = x.reshape(1, -1)
    y_predict = knn_clf.predict(X_predict)
    print(y_predict)
