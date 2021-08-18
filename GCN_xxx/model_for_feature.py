import torch.nn as nn
import torch.nn.functional as F
from GCN.layer import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, exfeature, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)   # gc1输入尺寸nfeat，输出尺寸nhid
        self.gc2 = GraphConvolution(nhid, exfeature)  # gc2输入尺寸nhid，输出尺寸exfeature
        self.dropout = dropout
        self.output = nn.Linear(exfeature, 2)


    # 输入分别是特征和邻接矩阵。最后输出为输出层做log_softmax变换的结果
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))    # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
        x = self.gc2(x, adj)
        out_class = self.output(x)
        # x输出尺寸为(nsamples,exfeature)
        return x, F.log_softmax(out_class, dim=1)

