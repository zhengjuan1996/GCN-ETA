from __future__ import division
from __future__ import print_function

import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import torch
import torch.nn.functional as F
from time import perf_counter
from GCN_ETA.utilsETA import load_data
np.random.seed(222)
'''数据预处理，代替GCN'''
'''在GCN的基础上去掉权重'''
def our_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        if i != degree-1:
            features = F.relu(torch.spmm(adj, features))
            # features = F.dropout(features, 0.5)
        elif i == degree-1:
            features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time


# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

exfeatures, precompute_time = our_precompute(features, adj, 2)

print("{:.4f}s".format(precompute_time))
print(exfeatures.size())


X = exfeatures
Y = labels

print("SGC+xgboost分类器")
scorings = ['accuracy', 'precision', 'recall', 'f1']
xgtime = []
for scoring in scorings:
    start = time.process_time()
    xgb = XGBClassifier(objective='binary:logitraw')
    scores = cross_val_score(xgb, X, Y, cv=5, scoring=scoring)  # cv为迭代次数。
    # print(scores)  # 打印输出每次迭代的度量值（准确度）
    end = time.process_time()
    xgtime.append(end - start)
    print(scoring + ": %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
print("SGC+xgboost分类器时间：{:.0f}flows/s" .format(X.shape[0]/(5*(np.mean(xgtime)+precompute_time))))

print('SGC+决策树分类器')
scorings = ['accuracy', 'precision', 'recall', 'f1']
dttime = []
for scoring in scorings:
    start = time.process_time()
    clf = DecisionTreeClassifier(criterion='gini')
    scores = cross_val_score(clf, X, Y, cv=5, scoring=scoring)  # cv为迭代次数。
    end = time.process_time()
    dttime.append(end - start)
    # print(scores)  # 打印输出每次迭代的度量值（准确度）
    print(scoring+": %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
print("SGC+决策树分类器时间：{:.0f}flows/s" .format(X.shape[0]/(5*(np.mean(dttime)+precompute_time))))
