import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time

start = time.clock()
# load train data

train = pd.read_csv('./data/train.csv')


# split feature and label

feature = train.drop('label', axis=1)
label = train[['label']]


# 设置一个阈值，为0-255之间，大于该值的设置为 1， 小于该值的设置为0

feature[feature > 0] = 1
feature[feature <= 0] = 0

# split train and test data

train_feature, test_feature, train_label, test_label = train_test_split(feature, label, test_size=0.2, random_state=0)

# fit classifier model

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(train_feature, train_label)


# predict test_feature

predict_test = clf.predict(test_feature)


# print result

print(metrics.classification_report(test_label, predict_test))


# predict test data,

test = pd.read_csv('./data/test.csv')


# 设置一个阈值，为0-255之间，大于该值的设置为 255， 小于该值的设置为0

test[test > 0] = 1
test[test < 0] = 0


# you need save file as .csv like sample_submission.csv

predict = clf.predict(test)
predict = pd.DataFrame(predict)
predict.to_csv('./data/predict.csv')
print("finish")



elapsed = (time.clock() - start)
print("Time used:", int(elapsed), "s")
