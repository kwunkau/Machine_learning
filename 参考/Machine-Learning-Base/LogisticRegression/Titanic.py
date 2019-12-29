import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.linear_model import stochastic_gradient

pd.set_option('display.max_columns', None)  # 输出结果显示全部列

# 读取数据
titanic = pd.read_csv("train.csv")
# 查看数据前5行
# print(data.head())
# 查看描述性统计,只能看数值型数据.
print(titanic.describe())

# 数据预处理
# 缺失值填充，Age列缺失的值，按中位数填充
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

print(titanic.describe())

# 把机器学习不能处理的字符值转换成机器学习可以处理的数值
# .loc 通过自定义索引获取数据 , 其中 .loc[:,:]中括号里面逗号前面的表示行，逗号后面的表示列
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# 通过统计三个登船地点人数最多的填充缺失值
titanic["Embarked"] = titanic["Embarked"].fillna("S")
# 字符处理
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# 选择我们要用到的特征
x = titanic[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = titanic[["Survived"]]

# 划分训练集和验证集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

LR = LogisticRegression()
sgdc=stochastic_gradient.SGDClassifier(loss="log", penalty="l2", max_iter=10000)

# 调用Logistic中的fit()函数来训练模型参数
LR.fit(x_train, y_train)

sgdc.fit(x_train, y_train)
# 使用训练好的模型LR 对 x_test进行预测，结果存储在变量x_pred中
LR_pred = LR.predict(x_test)
sgdc_pred = sgdc.predict(x_test)
# 使用学习器模型自带的score函数获得模型在测试集上的准确率
accuracy = LR.score(x_test, y_test)
print(accuracy)
# 利用classification_report获得其他三个评估指标
LR_report = classification_report(y_test, LR_pred, target_names=["存活", "死亡"])
print(LR_report)


sgdc_accuracy = sgdc.score(x_test, y_test)
print('随机梯度下降的分类准确率是：%f' % sgdc_accuracy)
print('利用classification_report获得其他三个评估指标')
sgdc_report = classification_report(y_test, sgdc_pred, target_names=["存活", "死亡"])
print(sgdc_report)