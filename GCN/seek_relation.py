import pandas as pd
import time
from sklearn.utils import shuffle

# 将两个分类随机抽样成1：1的比例
data = pd.read_csv(r"E:\工作\DataCon\total\csv\tuple_graph.csv")
data['label'] = data['label'].replace([0, 1], ['white', 'black'])
data0 = data[data['label'] == 'white']
data1 = data[data['label'] == 'black'].sample(n=data0.shape[0], random_state=123, axis=0)
data_train = pd.concat([data0, data1], axis=0)
data_train = shuffle(data_train)
data_train.to_csv(r"E:\工作\DataCon\total\csv\ETA.content.csv", index=False)

# 寻求训练集节点中的关联，两个流有公共IP即为有关联
data = pd.read_csv(r"E:\工作\DataCon\total\csv\ETA.content.csv")

edges1 = []
edges2 = []
src = data['ip.src']
dst = data['ip.dst']
ID = data['ID']
for i in range(data.shape[0]):
    # start = time.process_time()
    for j in range(data.shape[0]):
        if i != j and (set([src[j], dst[j]]) & set([src[i], dst[i]]) != set()):
            edges1.append(ID[i])
            edges2.append(ID[j])
    # end = time.process_time()
    # print('Running time: %s Seconds' % (end - start))
    print(i)
edges_unordered = pd.DataFrame({'ID': edges1, 'relative_ID': edges2})
edges_unordered.to_csv(r"E:\工作\DataCon\total\csv\ETA.adjacency.csv", index=False)

