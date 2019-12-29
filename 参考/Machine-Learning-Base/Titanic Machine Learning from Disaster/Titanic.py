import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import re
pd.set_option('display.max_columns', None)  # 输出结果显示全部列

# 导入数据
titanic = pd.read_csv("train.csv")
# print(titanic.head())  # 查看前几行数据，默认为前5行
print(titanic.describe())  # 查看描述性统计,只能看数值型数据.

#缺失值填充，Age列缺失的值，按中位数填充
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
#缺失值填充，Test集的Fare有一个缺失，按中位数来填充
print(titanic.describe())

#把机器学习不能处理的字符值转换成机器学习可以处理的数值
# .loc 通过自定义索引获取数据 , 其中 .loc[:,:]中括号里面逗号前面的表示行，逗号后面的表示列
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

#通过统计三个登船地点人数最多的填充缺失值
titanic["Embarked"] = titanic["Embarked"].fillna("S")
#字符处理
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

#选择我们要用到的特征
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
#将线性回归方法导进来
alg = LinearRegression()
#将m个样本平均分成11份进行交叉验证
kf = KFold(titanic.shape[0], n_folds=11, random_state=1)

predictions = []
for train, test in kf:
    #将predictors作为测试特征
    train_predictors = (titanic[predictors].iloc[train, :])
    #获取到数据集中交叉分类好的标签，即是否活了下来
    train_target = titanic["Survived"].iloc[train]
    #将数据放进去做训练
    alg.fit(train_predictors, train_target)
    #我们现在可以使用测试集来进行预测
    test_predictions = alg.predict(titanic[predictors].iloc[test, :])
    predictions.append(test_predictions)

#使用线性回归得到的结果是在区间[0,1]上的某个值，需要将该值转换成0或1
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

#查看模型准确率
accuracy = sum(predictions == titanic["Survived"]) / len(predictions)
print(accuracy)



#使用逻辑回归算法
alg = LogisticRegression(random_state=1)
#使用逻辑回归做交叉验证
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=11)
#取scores的平均值
print("逻辑回归")
print(scores)
print(scores.mean())


#使用随机森林

#指定随机森林的参数 n_estimators设置决策树的个数  min_samples_split最小的样本个数  min_samples_leaf 最小叶子节点的个数
alg = RandomForestClassifier(random_state=1, n_estimators=20, min_samples_split=4, min_samples_leaf=2)
kf = cross_validation.KFold(titanic.shape[0], n_folds=11, random_state=1)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
print(scores.mean())




#提取新的特征
#将船上所有人的亲属朋友关系加起来，新建一个特征，用来表示每个人的亲属关系
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
#根据名字长度组成一个新特征
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

#根据称呼来建立一个新特征
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.',name)

    if title_search:
        return title_search.group(1)
    return ""

titles = titanic["Name"].apply(get_title)
print(pd.value_counts(titles))
#将称号转换成数值表示
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 8, "Mlle": 9,
                 "Mme": 10, "Don": 11, "Lady": 12, "Countess": 13, "Jonkheer": 14, "Sir": 15, "Capt": 16, "Ms": 17
                }
for k, v in title_mapping.items():
    titles[titles == k] = v
#print(pd.value_counts(titles))

#添加Title特征
titanic["Title"] = titles

#下面进行特征选择

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "NameLength", "Title"]
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

scores = -np.log10(selector.pvalues_)

#画图看各个特征的重要程度
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

predictors = ["Pclass", "Sex", "Fare", "NameLength", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=20, min_samples_split=4, min_samples_leaf=2)

kf = cross_validation.KFold(titanic.shape[0], n_folds=11, random_state=1)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
print(scores.mean())

algorithms = [
    [RandomForestClassifier(random_state=1, n_estimators=20, min_samples_split=4, min_samples_leaf=2),
     ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'NameLength', 'Title']],
    [LogisticRegression(random_state=1),
     ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'NameLength', 'Title']]
]

kf = KFold(titanic.shape[0], n_folds=11, random_state=1)
predictions = []
for train, test in kf:
    train_target = titanic['Survived'].iloc[train]
    full_test_predictions = []
    for alg, predictors in algorithms:
        alg.fit(titanic[predictors].iloc[train, :], train_target)
        test_prediction = alg.predict_proba(titanic[predictors].iloc[test, :].astype(float))[:, 1]
        full_test_predictions.append(test_prediction)
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions > .5] = 1
    test_predictions[test_predictions <= .5] = 0
    predictions.append(test_predictions)
predictions = np.concatenate(predictions, axis=0)
accuracy = sum(predictions == titanic['Survived']) / len(predictions)  # 测试准确率
print(accuracy)

