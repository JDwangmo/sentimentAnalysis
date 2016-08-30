#coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import gensim
from sklearn.cluster import KMeans
import csv;

dataname = '/home/jdwang/PycharmProjects/sentimentAnalysis/dataset/cvaw1.csv'  # 训练集文件
vectormodel = 'wiki.zh.text.model'  # 模型文件
data = pd.read_csv(dataname,  # 读入训练集
               sep=',',
               encoding='utf8')


threshold = 0.5

def knnCluster(train,clusters_size):
    words = train[['Word_jian']].values
    filter_xy = train[['Valence_Mean', 'Arousal_Mean']].values
    random_seed = 1
    clf=KMeans(n_clusters=clusters_size, random_state=random_seed)
    class_arr =clf.fit_predict(filter_xy) # n_clusters 聚类个数

    clusters = [[] for i in range(clusters_size) ]
    #showChart(filter_xy,class_arr)
    for i in range(len(class_arr)):
        clusters[class_arr[i]].append(words[i][0])
    return (clusters,clf.cluster_centers_)

def data_split(data, ratio=0.8):
    '''
    数据集分割：按ratio比例将数据集分割成训练集和测试集
    :param data: 全体数据
    :param ratio: 分割比例
    :return: 训练集和测试集
    '''
    np.random.seed(1)  #进行多种方法比较的时候，不注释这一行，才能有比较性。
    size = int(len(data) * ratio)  # 训练集长度
    shuffle = range(len(data))  # 训练集索引
    np.random.shuffle(shuffle)  # 随机打乱索引

    train = data.iloc[shuffle[:size]]
    test = data.iloc[shuffle[size:]]
    return train, test

def calculateLoss(clusters,centers,word_VA):
	loss=0;
	for i in range(0,len(clusters)):
		for j in range(0,len(clusters[i])):
			loss+=(word_VA[clusters[i][j]][0]-centers[i][0])**2+(word_VA[clusters[i][j]][1]-centers[i][1])**2
	loss=loss/len(word_VA)
	return loss

def output(result):
    path='/home/jdwang/PycharmProjects/sentimentAnalysis/kmeans_knn/k-means_result.csv'
    csvfile=file(path,'wb')
    spamwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['cluster_size','loss']);
    for i in range(0,len(result)):
        spamwriter.writerow([result[i][0],result[i][1]]);

    csvfile.close()

def run():
	train, test = data_split(data)
	train_old = train[['Word_jian', 'Valence_Mean', 'Arousal_Mean']].values
	words=  train[['Word_jian']].values
	word_temp=[];
	for i in range(0,len(words)):
		word_temp.append(words[i][0])
	test_words = test[['No.', 'Word_jian', 'Valence_Mean', 'Arousal_Mean']].values
	test = test[['Valence_Mean', 'Arousal_Mean']].values
	word_VA=dict(zip(train['Word_jian'].values,train[['Valence_Mean', 'Arousal_Mean']].values ))
	result=[];
	clusters_size=[i for i in range(2,200) ]
	for i in range(0,len(clusters_size)):
		(clusters,centers) = knnCluster(train,clusters_size[i]) # 聚类结束后的点
		loss=calculateLoss(clusters,centers,word_VA)
		result.append([clusters_size[i],loss])

	output(result);





run();