import numpy as np
import pandas as pd

dataset = pd.read_csv('watermelon_3.csv', delimiter=",")
del dataset['编号']
print(dataset)
X = dataset.values[:, :-1]
m, n = np.shape(X)
for i in range(m):
    X[i, n - 1] = round(X[i, n - 1], 3)
    X[i, n - 2] = round(X[i, n - 2], 3)
y = dataset.values[:, -1]
columnName = dataset.columns
colIndex = {}
for i in range(len(columnName)):
    colIndex[columnName[i]] = i

Pmap = {}
kindsOfAttribute = {}  # kindsOfAttribute[0] = 3,因为有3种不同的类型的"色泽"
for i in range(n):
    kindsOfAttribute[i] = len(set(X[:, i]))
continuousPara = {}  # 记忆一些参数的连续数据,以避免重复计算

goodList = []
badList = []
for i in range(len(y)):
    if y[i] == '是':
        goodList.append(i)
    else:
        badList.append(i)

import math


def P(colID, attribute, C):  # P(colName=attribute|C) P(色泽=青绿|是)
    if (colID, attribute, C) in Pmap:
        return Pmap[(colID, attribute, C)]
    curJudgeList = []
    if C == '是':
        curJudgeList = goodList
    else:
        curJudgeList = badList
    ans = 0
    if colID >= 6:
        mean = 1
        std = 1
        if (colID, C) in continuousPara:
            curPara = continuousPara[(colID, C)]
            mean = curPara[0]
            std = curPara[1]
        else:
            curData = X[curJudgeList, colID]
            mean = curData.mean()
            std = curData.std()
            # print(mean,std)
            continuousPara[(colID, C)] = (mean, std)
        ans = 1 / (math.sqrt(math.pi * 2) * std) * math.exp((-(attribute - mean) ** 2) / (2 * std * std))
    else:
        for i in curJudgeList:
            if X[i, colID] == attribute:
                ans += 1
        ans = (ans + 1) / (len(curJudgeList) + kindsOfAttribute[colID])
    Pmap[(colID, attribute, C)] = ans
    # print(ans)
    return ans


def predictOne(single):
    ansYes = math.log2((len(goodList) + 1) / (len(y) + 2))
    ansNo = math.log2((len(badList) + 1) / (len(y) + 2))
    for i in range(len(single)):  # 书上是连乘，但在实践中要把“连乘”通过取对数的方式转化为“连加”以避免数值下溢
        ansYes += math.log2(P(i, single[i], '是'))
        ansNo += math.log2(P(i, single[i], '否'))
    # print(ansYes,ansNo,math.pow(2,ansYes),math.pow(2,ansNo))
    if ansYes > ansNo:
        return '是'
    else:
        return '否'


def predictAll(iX):
    predictY = []
    for i in range(m):
        predictY.append(predictOne(iX[i]))
    return predictY


predictY = predictAll(X)
print(y)
print(np.array(predictAll(X)))

confusionMatrix = np.zeros((2, 2))
for i in range(len(y)):
    if predictY[i] == y[i]:
        if y[i] == '否':
            confusionMatrix[0, 0] += 1
        else:
            confusionMatrix[1, 1] += 1
    else:
        if y[i] == '否':
            confusionMatrix[0, 1] += 1
        else:
            confusionMatrix[1, 0] += 1
print(confusionMatrix)
