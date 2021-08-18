from __future__ import division
from __future__ import print_function

import time
import argparse  # argparse 是python自带的命令行参数解析包，可以用来方便地读取命令行参数
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from GCN.utilsETA import load_data, pinggu

# GCN
from GCN.models import GCN
from sklearn.model_selection import StratifiedKFold, train_test_split


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=30,
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
adj, features, labels, _, _, _ = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


def train(epoch):
    t = time.time()  # 返回当前时间
    model.train()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train]) #GCN
    acc_train, precision_train, recall_train, f1_train = pinggu(output[idx_train], labels[idx_train])  #计算准确率
    loss_train.backward()  # 反向求导  Back Propagation
    optimizer.step()  # 更新所有的参数  Gradient Descent

    if not args.fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])    # 验证集的损失函数
    acc_val, precision_val, recall_val, f1_val = pinggu(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'precision_train: {:.4f}'.format(precision_train.item()),
          'recall_train: {:.4f}'.format(recall_train.item()),
          'f1_train: {:.4f}'.format(f1_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'precision_val: {:.4f}'.format(precision_val.item()),
          'recall_val: {:.4f}'.format(recall_val.item()),
          'f1_val: {:.4f}'.format(f1_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

# 定义测试函数，相当于对已有的模型在测试集上运行对应的loss与accuracy
def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    # acc_test = accuracy(output[idx_test], labels[idx_test])
    acc_test, precision_test, recall_test, f1_test = pinggu(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "precision= {:.4f}".format(precision_test.item()),
          "recall= {:.4f}".format(recall_test.item()),
          "f1= {:.4f}".format(f1_test.item()))
    return acc_test, precision_test, recall_test, f1_test


# 5折交叉验证
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=322)
acc_tests, precision_tests, recall_tests, f1_tests = [], [], [], []
t_total = []
for train1, i in kfold.split(range(features.shape[0]), labels):
    idx_test = list(set(range(features.shape[0])).difference(set(train1)))
    idx_train, idx_val, _, _ = train_test_split(train1, labels[train1], random_state=11, test_size=0.1)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    # 数据写入cuda，便于后续加速
    if args.cuda:
        model.cuda()  # . cuda()会分配到显存里（如果gpu可用）
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    t_start = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    t_end = time.time()
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(t_end - t_start))
    t_total.append(t_end - t_start)
    acc_test, precision_test, recall_test, f1_test = test()
    acc_tests.append(acc_test)
    precision_tests.append(precision_test)
    recall_tests.append(recall_test)
    f1_tests.append(f1_test)
print("Test set results:")
print("accuracy: %0.4f (+/- %0.4f)" % (np.mean(acc_tests), np.std(acc_tests)))
print("precision: %0.4f (+/- %0.4f)" % (np.mean(precision_tests), np.std(precision_tests)))
print("recall: %0.4f (+/- %0.4f)" % (np.mean(recall_tests), np.std(recall_tests)))
print("f1: %0.4f (+/- %0.4f)" % (np.mean(f1_tests), np.std(f1_tests)))
print("每秒检测流数: %0.0f " % (len(idx_test)/np.mean(t_total)))
