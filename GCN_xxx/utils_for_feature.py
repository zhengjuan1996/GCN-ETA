import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import torch.nn as nn
from sklearn.metrics import recall_score, f1_score, accuracy_score
'''
先将所有由字符串表示的标签数组用set保存，set的重要特征就是元素没有重复，
因此表示成set后可以直接得到所有标签的总数，随后为每个标签分配一个编号，创建一个单位矩阵，
单位矩阵的每一行对应一个one-hot向量，也就是np.identity(len(classes))[i, :]，
再将每个数据对应的标签表示成的one-hot向量，类型为numpy数组
'''

def encode_onehot(labels):
    classes = set(labels)  # set() 函数创建一个无序不重复元素集
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in  # identity创建方矩阵
                    enumerate(classes)}     # 字典 key为label的值，value为矩阵的每一行
    # enumerate函数用于将一个可遍历的数据对象组合为一个索引序列
    labels_onehot = np.array(list(map(classes_dict.get, labels)),  # get函数得到字典key对应的value
                             dtype=np.int32)
    return labels_onehot
    # map() 会根据提供的函数对指定序列做映射
    # 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表
    #  map(lambda x: x ** 2, [1, 2, 3, 4, 5])
    #  output:[1, 4, 9, 16, 25]


def load_data(path='E:/工作/DataCon/total/csv/两类节点平衡（总共五万多）/', dataset="ETA"):
    '''Load  dataset'''
    print('Loading{} dataset...'.format(dataset))

    #读取ETA.content.csv数据，ETA.content.csv数据中包含每个流的特征，及每个流的分类
    '''ETA.content.csv共有53002行，每一行代表一个样本点，即一个流。每一行由三部分组成，
    分别是流的编号；流的特征，'in_packets', 'in_bytes', 'out_packets', 'out_bytes', 'duration'；流的的类别，black或者white'''

    idx_features_labels = pd.read_csv('{}{}.content.csv'.format(path,dataset))[['ID', 'in_packets', 'in_bytes', 'out_packets', 'out_bytes', 'duration', 'label']]
    # return idx_features_labels
    # in_bytes,out_bytes单位转换为kB
    in_bytes = idx_features_labels['in_bytes']
    out_bytes = idx_features_labels['out_bytes']
    in_bytes_deal = []
    out_bytes_deal = []
    for i in in_bytes:
        if 'bytes' in i:
            in_bytes_deal.append(int(i[0:-5])/1024)
        elif 'kB' in i:
            in_bytes_deal.append(int(i[0:-2]))

    for i in out_bytes:
        if 'bytes' in i:
            out_bytes_deal.append(int(i[0:-5])/1024)
        elif 'kB' in i:
            out_bytes_deal.append(int(i[0:-2]))

    idx_features_labels['in_bytes'] = in_bytes_deal
    idx_features_labels['out_bytes'] = out_bytes_deal

    #提取出特征和标签
    features = sp.csr_matrix(idx_features_labels[['in_packets', 'in_bytes', 'out_packets', 'out_bytes', 'duration']], dtype=np.float32) #把特征储存为csr型稀疏矩阵

    labels = encode_onehot(idx_features_labels['label'])
    # labels = idx_features_labels['label']

    #建图
    #cites file的每一行格式为：<flow ID> <relative_flow_ID>
    #根据前面的contents与这里的cites创建图，算出edges（度）矩阵与adj（邻接）矩阵
    # 取出流的id
    idx = np.array(idx_features_labels['ID'], dtype=np.int32)
    # idx_map = id：编号
    idx_map = {j: i for i, j in enumerate(idx)}
    # return idx_map
    # 由于文件中节点并非是按顺序排列的，因此建立一个编号为0-（node_size-1)的哈希表idx_map,
    # 哈希表中每一项为id.number，即节点id对应的编号为number
    '''cora.cites每一行有两个流编号，两个流之间有关联（有公共IP）'''
    edges_unordered = pd.read_csv('{}{}.adjacency.csv'.format(path, dataset))
    edges_unordered = np.array(edges_unordered.values, dtype=np.int32)
    # edges_unordered为直接从边表文件中直接读取的结果，是一个（edge_num,2)的数组，每一行标识一条边两个端点的idx
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), #flatten：降维，返回一维数组
                     dtype=np.int32).reshape(edges_unordered.shape)
    # edges_unoedered中存储的是端点id，要将每一项的id换成编号
    # 在idx_map中以idx作为键查找得到对应节点的编号，reshape成与edges_unoredered形状一样的数组
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # adj
    '''
    (163, 402)	1.0
    (163, 659)	1.0
    (163, 1696)	1.0
    (163, 2295)	1.0
    (163, 1274)	1.0
    :	:
    (1887, 2258)	1.0
    (1902, 1887)	1.0
    (837, 1686)	1.0
    '''
    # 根据coo矩阵性质，这一段的作用就是，网络有多少条边，邻接矩阵就有多少个1，
    # 所以先创建一个长度为edge_num的全1 数组，每个1的填充位置就是一条边中两个端点的编号，
    # 即edges[:,0],edges[:,1],矩阵的形状为(node_size,node_size).
    #build symmetric adjacency matrix  论文里A^=(D~)^0.5 A~ (D~)^0.5这个公式
    adj = adj+adj.T.multiply(adj.T > adj)-adj.multiply(adj.multiply(adj.T > adj))
    # 对于无向图,邻接矩阵是对称.上一步得到的adj是按有向图构建的,转换为无向图的邻接矩阵需要扩充成对称矩阵
    features = normalize(features)
    # 自己定义的normalize规则
    adj = normalize(adj+sp.eye(adj.shape[0]))   #eye创建单位矩阵,第一个参数为行数,第二个为列数
    # 对应公式A~=A+IN
    labels = torch.LongTensor(np.where(labels)[1])

    features = torch.FloatTensor(np.array(features.todense()))  # tensor为pytorch常用的数据结构
    #这里将onthot label转回index
    adj = sparse_mx_to_torch_sparse_tensor(adj)   # 邻接矩阵转为tensor处理

    return adj, features, labels

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels) # 使用type_as(tesnor)将张量转换为给定类型的张量。
    correct = preds.eq(labels).double()  # 记录等于preds的label eq:equal
    correct = correct.sum()
    return correct / len(labels)

def pinggu(output, labels):
    # acc = accuracy(output, labels)
    labels = np.array(labels.cpu())
    output = np.array(output.max(1)[1].cpu())
    acc = accuracy_score(labels, output)
    r = recall_score(labels, output)
    f1 = f1_score(labels, output)
    return acc, r, f1

def pinggu_mlp(output, labels):
    # acc = accuracy(output, labels)
    # labels = np.array(labels.cpu())
    # output = np.array(output.max(1)[1].cpu())
    acc = accuracy_score(labels, output)
    r = recall_score(labels, output)
    f1 = f1_score(labels, output)
    return acc, r, f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):    # 把一个sparse matrix转为torch稀疏张量
    """
    numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    pytorch中的tensor转化成numpy中的ndarray : numpy()
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # 不懂的可以去看看COO性稀疏矩阵的结构
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
