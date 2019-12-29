from math import log
import operator

def calcShannonEnt(dataSet):  # 计算数据的熵(entropy)
    numEntries=len(dataSet)  # 数据条数
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1] # 每行数据的最后一个字（类别）
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1  # 统计有多少个类以及每个类的数量
    shannonEnt=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries # 计算单个类的熵值
        shannonEnt-=prob*log(prob,2) # 累加每个类的熵值
    return shannonEnt

def createDataSet1():    # 创造示例数据
     dataSet = [['青绿' , '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
               ['乌黑' , '蜷缩' , '沉闷' , '清晰' , '凹陷' , '硬滑' , '好瓜'] ,
               ['乌黑' , '蜷缩' , '浊响' , '清晰' , '凹陷' , '硬滑' , '好瓜'] ,
               ['青绿' , '蜷缩' , '沉闷' , '清晰' , '凹陷' , '硬滑' , '好瓜'] ,
               ['浅白' , '蜷缩' , '浊响' , '清晰' , '凹陷' , '硬滑' , '好瓜'] ,
               ['青绿' , '稍缩' , '浊响' , '清晰' , '稍凹' , '软粘' , '好瓜'] ,
               ['乌黑' , '稍缩' , '浊响' , '稍糊' , '稍凹' , '软粘' , '好瓜'] ,
               ['乌黑' , '稍缩' , '浊响' , '清晰' , '稍凹' , '硬滑' , '好瓜'] ,
               ['乌黑' , '稍缩' , '沉闷' , '稍糊' , '稍凹' , '硬滑' , '好瓜'] ,
               ['青绿' , '硬挺' , '清脆' , '清晰' , '平坦' , '硬滑' , '坏瓜'] ,
               ['浅白' , '硬挺' , '清脆' , '模糊' , '平坦' , '软粘' , '坏瓜'] ,
               ['浅白' , '蜷缩' , '浊响' , '模糊' , '平坦' , '硬滑' , '坏瓜'] ,
               ['青绿' , '稍缩' , '浊响' , '稍糊' , '凹陷' , '软粘' , '坏瓜'] ,
               ['浅白' , '稍缩' , '沉闷' , '稍糊' , '凹陷' , '硬滑' , '坏瓜'] ,
               ['乌黑' , '稍缩' , '浊响' , '清晰' , '稍凹' , '软粘' , '坏瓜'] ,
               ['浅白' , '蜷缩' , '浊响' , '模糊' , '平坦' , '硬滑' , '坏瓜'] ,
               ['青绿' , '蜷缩' , '沉闷' , '稍糊' , '稍凹' , '硬滑' , '坏瓜'] ]
    labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']  #6个特征
    return dataSet,labels

def splitDataSet(dataSet,axis,value): # 按某个特征分类后的数据
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec =featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):  # 选择最优的分类特征
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)  # 原始的熵
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob =len(subDataSet)/float(len(dataSet))
            newEntropy +=prob*calcShannonEnt(subDataSet)  # 按特征分类后的熵
        infoGain = baseEntropy - newEntropy  # 原始熵与按特征分类后的熵的差值
        if (infoGain>bestInfoGain):   # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
            bestInfoGain=infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):    #按分类后类别数量排序，比如：最后分类为2好瓜1坏瓜，则判定为好瓜；
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]  # 类别：好瓜或坏瓜
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet) #选择最优特征
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}} #分类结果以字典形式保存
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet\
                            (dataSet,bestFeat,value),subLabels)
    return myTree


if __name__=='__main__':
    dataSet, labels=createDataSet1()  # 创造示列数据
    print(createTree(dataSet, labels))  # 输出决策树模型结果
