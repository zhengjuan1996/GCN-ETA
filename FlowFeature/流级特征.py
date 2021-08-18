#coding=utf-8
import scipy.sparse as sp
import os
import pandas as pd
from FlowFeature import global_path
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from collections import Counter
import time
from sklearn.model_selection import cross_val_score  # 交叉验证所需的函数
from sklearn.ensemble import RandomForestClassifier
np.random.seed(222)
'''
完成流级特征分类功能
def tcp_tuple 提取tcp四元组数据并写入csv文件
'''

def tcp_tuple(path):
    '''
    指定tshark -r file.pcap -q -z conv,tcp命令，得到tcp四元组的统计信息
    处理命令返回结果的字符串，去掉不需要的内容
    将字符串分割为符合csv文件的格式，写入DataFrame
    最后统一输入csv
    :param path: 存储用于训练的pcap文件的文件夹
    :return DataFrame: 四元组数据组成的DataFrame句柄
    '''
    files = os.listdir(path)
    # os.popen('D:')
    # os.popen('cd D:\Program Files\Wireshark')
    cmd = 'tshark -r {pcap} -q -z conv,tcp'
    # headers = [['ip.src', 'ip.dst', 'in_packets', 'in_bytes', 'out_packets', 'out_bytes', 'duration']]
    headers = [['ip.src', 'ip.dst', 'in_packets', 'in_bytes', 'out_packets', 'out_bytes', 'duration']]

    sum_df = pd.DataFrame(headers)

    '''
    获取所有pcap文件内的四元组数据
    '''
    for file in files:
        pcap = path + '\\' + file
        data = os.popen(cmd.format(pcap=pcap)).read()   #获取命令执行结果
        data = np.array(data.split('|')[-1].split('=')[0].split())  #切割出需要的数据内容
        df = pd.DataFrame(data.reshape(-1, 11)) #重新排列，转为csv格式
        df = df.drop([1, 7, 8, 9], axis=1)  #筛除无用数据
        df = df.T.reset_index(drop=True).T  #重置列索引
        sum_df = pd.concat([sum_df, df])    #拼接进总的DataFrame
        print(file, ' 提取完毕')

    return sum_df


def tuple_csv(DataFrame, path):
    '''
    将数据打上类别标签，写入csv文件
    :param DataFrame: path路径下所有的pcap文件中提取出来的tcp流元组数据
    :return:
    '''
    label_white = ['label']
    label_black = ['label']

    if path == global_path.white_path:
        for i in range(DataFrame.shape[0] -1):
            label_white.append('0')
        DataFrame['7'] = label_white
    elif path == global_path.black_path:
        for i in range(DataFrame.shape[0] -1):
            label_black.append('1')
        DataFrame['7'] = label_black

    DataFrame = DataFrame.drop([0], axis=0)
    DataFrame.columns = ['ip.src', 'ip.dst', 'in_packets', 'in_bytes', 'out_packets', 'out_bytes', 'duration', 'label']
    print(DataFrame)
    if os.path.exists(tuple_data_path):
        DataFrame.to_csv(tuple_data_path, index=False, header=False, mode='a')
    else:
        DataFrame.to_csv(tuple_data_path, index=False)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx

def classify(DataFrame):
    '''
    读取DataFrame，训练分类器
    预测测试数据类型
    :param DataFrame: 所有pcap文件中提取出来的tcp流元组数据
    :return:
    '''
    # in_bytes,out_bytes单位转换为kB
    in_bytes = DataFrame['in_bytes']
    out_bytes = DataFrame['out_bytes']
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

    DataFrame['in_bytes'] = in_bytes_deal
    DataFrame['out_bytes'] = out_bytes_deal


    X = sp.csr_matrix(DataFrame[['in_packets', 'in_bytes', 'out_packets', 'out_bytes', 'duration']], dtype=np.float32)
    X = normalize(X)
    Y = DataFrame['label'].replace(['black', 'white'], [1, 0])
    print(Counter(Y).items())

    print("xgboost分类器")
    scorings = ['accuracy', 'precision', 'recall', 'f1']
    xgtime = []
    for scoring in scorings:
        start = time.process_time()
        xgb = XGBClassifier()
        scores = cross_val_score(xgb, X, Y, cv=5, scoring=scoring)  # cv为迭代次数。
        # print(scores)  # 打印输出每次迭代的度量值（准确度）
        end = time.process_time()
        xgtime.append(end - start)
        print(scoring+": %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print("xgboost分类器时间：{:.0f}flows/s".format(X.shape[0] / (5 * np.mean(xgtime))))


    print('决策树分类器')
    dttime = []
    scorings = ['accuracy', 'precision', 'recall', 'f1']
    for scoring in scorings:
        start = time.process_time()
        clf = DecisionTreeClassifier()
        scores = cross_val_score(clf, X, Y, cv=5, scoring=scoring)  # cv为迭代次数。
        # print(scores)  # 打印输出每次迭代的度量值（准确度）
        end = time.process_time()
        dttime.append(end - start)
        print(scoring+": %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print("决策树分类器时间：{:.0f}flows/s".format(X.shape[0] / (5 * np.mean(dttime))))

    print('逻辑回归分类器')
    lrtime = []
    scorings = ['accuracy', 'precision', 'recall', 'f1']
    for scoring in scorings:
        start = time.process_time()
        lr = LogisticRegression(solver='liblinear', C=1)
        scores = cross_val_score(lr, X, Y, cv=5, scoring=scoring)  # cv为迭代次数。
        # print(scores)  # 打印输出每次迭代的度量值（准确度）
        end = time.process_time()
        lrtime.append(end - start)
        print(scoring+": %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print("逻辑回归分类器时间：{:.0f}flows/s".format(X.shape[0] / (5 * np.mean(lrtime))))

    print('朴素贝叶斯分类器')
    bltime = []
    scorings = ['accuracy', 'precision', 'recall', 'f1']
    for scoring in scorings:
        start = time.process_time()
        bl = BernoulliNB(alpha=1e-05, binarize=0.1)
        scores = cross_val_score(bl, X, Y, cv=5, scoring=scoring)  # cv为迭代次数。
        # print(scores)  # 打印输出每次迭代的度量值（准确度）
        end = time.process_time()
        bltime.append(end - start)
        print(scoring+": %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print("朴素贝叶斯分类器时间：{:.0f}flows/s".format(X.shape[0] / (5 * np.mean(bltime))))

    print('随机森林分类器')
    rftime = []
    scorings = ['accuracy', 'precision', 'recall', 'f1']
    for scoring in scorings:
        start = time.process_time()
        rf = RandomForestClassifier()
        scores = cross_val_score(rf, X, Y, cv=5, scoring=scoring)  # cv为迭代次数。
        # print(scores)  # 打印输出每次迭代的度量值（准确度）
        end = time.process_time()
        rftime.append(end - start)
        print(scoring+": %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print("随机森林分类器时间：{:.0f}flows/s".format(X.shape[0] / (5 * np.mean(rftime))))

    print('KNN分类器')
    kntime = []
    scorings = ['accuracy', 'precision', 'recall', 'f1']
    for scoring in scorings:
        start = time.process_time()
        kn = KNeighborsClassifier(5)
        scores = cross_val_score(kn, X, Y, cv=5, scoring=scoring)  # cv为迭代次数。
        # print(scores)  # 打印输出每次迭代的度量值（准确度）
        end = time.process_time()
        kntime.append(end - start)
        print(scoring+": %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print("KNN分类器时间：{:.0f}flows/s".format(X.shape[0] / (5 * np.mean(kntime))))

if __name__ == '__main__':
    tuple_data_path = global_path.csv_path + '\\ETA.content.csv'
    if not os.path.exists(tuple_data_path):     #若文件不存在，需要读取pcap文件，建立训练数据集
        white_df = tcp_tuple(global_path.white_path)
        tuple_csv(white_df, global_path.white_path)

        black_df = tcp_tuple(global_path.black_path)
        tuple_csv(black_df, global_path.black_path)

    df = pd.read_csv(tuple_data_path)
    print(df['label'].value_counts())
    classify(df)