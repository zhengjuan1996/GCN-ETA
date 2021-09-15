from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import torch.nn.functional as F
import torch
import torch.optim as optim

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier

from GCN_xxx.utils_for_feature import load_data, pinggu
from GCN_xxx.model_for_feature import GCN
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

# 如果程序不禁止使用gpu且当前主机的gpu可用，arg.cuda就为True
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)   # 为CPU设置种子用于生成随机数，以使得结果是确定的
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels = load_data()

# gcn的训练模块
def train(epoch):
    t = time.time()  # 返回当前时间
    model.train()
    optimizer.zero_grad()
    output_feature, output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train]) #GCN
    acc_train, recall_train, f1_train = pinggu(output[idx_train], labels[idx_train])  #计算准确率
    loss_train.backward()  # 反向求导  Back Propagation
    optimizer.step()  # 更新所有的参数  Gradient Descent


    if not args.fastmode:
        model.eval()
        output_feature, output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])    # 验证集的损失函数
    acc_val, recall_val, f1_val = pinggu(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'recall_train: {:.4f}'.format(recall_train.item()),
          'f1_train: {:.4f}'.format(f1_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'recall_val: {:.4f}'.format(recall_val.item()),
          'f1_val: {:.4f}'.format(f1_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return output_feature

# 定义测试函数，相当于对已有的模型在测试集上运行对应的loss与accuracy
def test():
    model.eval()
    output_feature, output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test, recall_test, f1_test = pinggu(output[idx_test], labels[idx_test])
    return acc_test, recall_test, f1_test



# train
for exfeature in [7]:
# for exfeature in range(5,20):
    # Model and optimizer
    #训练GCN
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                exfeature=exfeature,  #提取过后的特征数量
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    # 数据写入cuda，便于后续加速
    if args.cuda:
        model.cuda()   # . cuda()会分配到显存里（如果gpu可用）
        features = features.cuda()
        adj = adj.cuda()

    idx_train1, idx_test, _, _ = train_test_split(range(features.shape[0]), labels, random_state=1, test_size=0.8)
    idx_train, idx_val, _, _ = train_test_split(idx_train1, labels[idx_train1], random_state=2, test_size=0.1)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    t_start = time.time()
    for epoch in range(args.epochs):
        output_feature = train(epoch)
    t_end = time.time()
    print("提取特征数为：" + str(exfeature))
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(t_end - t_start))
    acc_test, recall_test, f1_test = test()
    print("Test set results:")
    print("accuracy:" + str(acc_test))
    print("recall:" + str(recall_test))
    print("f1:" + str(f1_test))

    # 训练ML
    exfeature_times = t_end - t_start
    Y = labels[idx_test]
    X = output_feature[idx_test].cpu().detach().numpy()
    print(X.shape)
    # print(Counter(Y).items())
    print("提取特征数为："+str(exfeature))
    print("GCN+xgboost分类器")
    scorings = ['accuracy', 'precision', 'recall', 'f1']
    xgtime = []
    for scoring in scorings:
        start = time.process_time()
        xgb = XGBClassifier(objective='binary:logitraw')
        scores = cross_val_score(xgb, X, Y, cv=5, scoring=scoring)  # cv为迭代次数。
        # print(scores)  # 打印输出每次迭代的度量值（准确度）
        end = time.process_time()
        xgtime.append((end - start)/5)
        print(scoring + ": %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print("GCN+xgboost分类器速度：{:.0f}flows/s" .format(X.shape[0]/(5*(np.mean(xgtime)+exfeature_times))))

    print('GCN+决策树分类器')
    scorings = ['accuracy', 'precision', 'recall', 'f1']
    dttime = []
    for scoring in scorings:
        start = time.process_time()
        clf = DecisionTreeClassifier()
        scores = cross_val_score(clf, X, Y, cv=5, scoring=scoring)  # cv为迭代次数。
        # print(scores)  # 打印输出每次迭代的度量值（准确度）
        end = time.process_time()
        dttime.append((end - start)/5)
        print(scoring+": %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print("GCN+决策树分类器速度：{:.0f}flows/s".format(X.shape[0] / (5 * (np.mean(dttime) + exfeature_times))))

    print('GCN+随机森林分类器')
    scorings = ['accuracy', 'precision', 'recall', 'f1']
    rftime = []
    for scoring in scorings:
        start = time.process_time()
        rf = RandomForestClassifier()
        scores = cross_val_score(rf, X, Y, cv=5, scoring=scoring)  # cv为迭代次数。
        # print(scores)  # 打印输出每次迭代的度量值（准确度）
        end = time.process_time()
        rftime.append((end - start) / 5)
        print(scoring + ": %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print("GCN+随机森林分类器速度：{:.0f}flows/s".format(X.shape[0] / (5 * (np.mean(rftime) + exfeature_times))))

    print('GCN+贝叶斯分类器')
    scorings = ['accuracy', 'precision', 'recall', 'f1']
    bltime = []
    for scoring in scorings:
        start = time.process_time()
        bl = BernoulliNB(alpha=1e-05, binarize=0.1)
        scores = cross_val_score(bl, X, Y, cv=5, scoring=scoring)  # cv为迭代次数。
        # print(scores)  # 打印输出每次迭代的度量值（准确度）
        end = time.process_time()
        bltime.append((end - start) / 5)
        print(scoring + ": %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print("GCN+贝叶斯分类器速度：{:.0f}flows/s".format(X.shape[0] / (5 * (np.mean(bltime) + exfeature_times))))

    print('GCN+逻辑回归分类器')
    scorings = ['accuracy', 'precision', 'recall', 'f1']
    lrtime = []
    for scoring in scorings:
        start = time.process_time()
        lf = LogisticRegression(solver='liblinear', C=1)
        scores = cross_val_score(lf, X, Y, cv=5, scoring=scoring)  # cv为迭代次数。
        # print(scores)  # 打印输出每次迭代的度量值（准确度）
        end = time.process_time()
        lrtime.append((end - start) / 5)
        print(scoring + ": %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print("GCN+逻辑回归分类器速度：{:.0f}flows/s".format(X.shape[0] / (5 * (np.mean(lrtime) + exfeature_times))))

    print('GCN+knn分类器')
    scorings = ['accuracy', 'precision', 'recall', 'f1']
    kntime = []
    for scoring in scorings:
        start = time.process_time()
        kn = KNeighborsClassifier(5)
        scores = cross_val_score(kn, X, Y, cv=5, scoring=scoring)  # cv为迭代次数。
        # print(scores)  # 打印输出每次迭代的度量值（准确度）
        end = time.process_time()
        kntime.append((end - start) / 5)
        print(scoring + ": %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print("GCN+knn分类器速度：{:.0f}flows/s".format(X.shape[0] / (5 * (np.mean(kntime) + exfeature_times))))


