from sklearn.ensemble import RandomForestClassifier

# 建立随机森林分类器
random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
#n_estimators : 指森林中树的个数
#n_jobs : 拟合（fit）和预测（predict）时并行运行的job数目，当设置为-1时，job数被设置为核心（core）数。
#  训练数据集
random_forest.fit(train, train_labels)
#verbose :冗余控制 控制树增长过程中的冗余（verbosity）。​
# 提取重要特征
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

# 对测试数据进行预测
predictions = random_forest.predict_proba(test)[:, 1]