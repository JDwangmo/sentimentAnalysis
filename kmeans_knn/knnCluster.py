# coding:utf-8

import numpy as np
import gensim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

data_name = '/home/jdwang/PycharmProjects/sentimentAnalysis/dataset/cvaw1.csv'  # 训练集文件
vector_model = '/home/jdwang/PycharmProjects/sentimentAnalysis/dataset/word2vec_model_file/wiki.zh.text.model'  # 模型文件




data = pd.read_csv(data_name,  # 读入训练集
                   sep=',',
                   encoding='utf8')
model = gensim.models.Word2Vec.load(vector_model)
threshold = 0.5


def data_split(data, ratio=0.8):
    """
    数据集分割：按ratio比例将数据集分割成训练集和测试集

    :param data: 全体数据
    :param ratio: 分割比例
    :return: 训练集和测试集
    """

    np.random.seed(1)  # 进行多种方法比较的时候，不注释这一行，才能有比较性。
    size = int(len(data) * ratio)  # 训练集长度
    shuffle = range(len(data))  # 训练集索引
    np.random.shuffle(shuffle)  # 随机打乱索引

    train = data.iloc[shuffle[:size]]
    test = data.iloc[shuffle[size:]]
    return train, test


# 预测0。对同一个簇中，达到阈值才产生贡献。
def predict(word, word_VA, mean, clusters):
    '''
    :param word:unseen词
    :param word_VA: 训练词对应的VA值
    :param clusters: unseen所在的簇
    :return:
    '''

    if (len(clusters) == 1):  # 如果unseen因为不在模型中，或者训练词不足以预测出该词，就返回全体训练词的均值。
        return mean, False

    weighted_VA, totle_sims = 0, 0
    for word2 in clusters:  # 能在同一个簇中就证明能够加边，也就证明了在model中。
        sim = model.similarity(word, word2)
        if sim < threshold:
            continue
        tmp = np.array(word_VA[word])
        weighted_VA += tmp * sim
        totle_sims += sim
    result = weighted_VA / totle_sims
    return result, True


def showChart(points, position):  # 绘制图表
    plt.scatter(points[:, 0], points[:, 1], c=position)  # scatter绘制散点
    plt.show()
    return


def knnCluster(train):
    words = train[['Word_jian']].values
    filter_xy = train[['Valence_Mean', 'Arousal_Mean']].values
    random_seed = 1
    clusters_size = 5
    class_arr = KMeans(n_clusters=clusters_size, random_state=random_seed).fit_predict(filter_xy)  # n_clusters 聚类个数
    clusters = [[] for i in range(clusters_size)]
    showChart(filter_xy,class_arr)
    for i in range(len(class_arr)):
        clusters[class_arr[i]].append(words[i])
    return clusters


def run():
    train, test = data_split(data)
    # print test.head()
    # quit()
    train_old = train[['Word_jian', 'Valence_Mean', 'Arousal_Mean']].values
    test_words = test[['No.', 'Word_jian', 'Valence_Mean', 'Arousal_Mean']].values
    test = test[['Valence_Mean', 'Arousal_Mean']].values
    clusters = knnCluster(train)  # 聚类结束后的点


run()
