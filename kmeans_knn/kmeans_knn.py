# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-27'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from dataset.data_util import DataUtil

dutil = DataUtil()


def showChart(points, position):  # 绘制图表
    plt.scatter(points[:, 0], points[:, 1], c=position)  # scatter绘制散点
    plt.show()
    return

def knnCluster(train,num_clusters=5,verbose=0):
    # words = train[['Word_jian']].values
    filter_xy = train[['Valence_Mean', 'Arousal_Mean']].values
    random_seed = 1
    class_arr = KMeans(n_clusters=num_clusters, random_state=random_seed).fit_predict(filter_xy)  # n_clusters 聚类个数
    if verbose>1:
        showChart(filter_xy,class_arr)

    # clusters = [[] for i in range(num_clusters)]
    # for i in range(len(class_arr)):
    #     clusters[class_arr[i]].append(words[i])
    # 将聚类结果保存
    train.loc[:,'Cluster_ID'] = class_arr
    # return clusters

def predict(train_data,word,top_word = 3,top_cluster=3,verbose=0):

    cluster_scores = []
    # print(train_data.head())
    for cluster_id,group in train_data.groupby(by=['Cluster_ID']):
        # print('cluster %d: %d samples'%(cluster_id,len(group)))
        # 计算相似性
        words_sim = np.asarray([dutil.get_word_similarity(word,item) for item in group['Word_jian'].values])
        # 获取相似性降序的索引
        sorted_index = np.argsort(words_sim)[-1::-1]
        # print(sorted_index)

        top_words = group[['Word_jian','Valence_Mean', 'Arousal_Mean']].values[sorted_index[:top_word]]
        top_words_sim = words_sim[sorted_index[:top_word]]
        score = np.mean(words_sim[sorted_index[:top_word]])
        # print(group.iloc[sorted_index])
        # print(top_words_sim)
        # [print(item) for item in top_words]
        # print(score)
        cluster_scores.append([cluster_id,score,top_words,top_words_sim])
    # print(cluster_scores)
    # 根据分数升序排序
    sorted_cluster_scores =sorted(cluster_scores,key=lambda x:x[1],reverse=True)
    if verbose>1:
        print_array(sorted_cluster_scores)

    V,A = get_predict_result(sorted_cluster_scores,top=top_cluster)

    return V,A

def get_predict_result(scores,top=1):
    """
        根据近似性结果，去除top3的词，进行算 VA值

    :param scores:
    :param top:
    :return:
    """
    scores = scores[:top]
    V = []
    A = []
    sim = []
    for item in scores:
        V.extend([i[1] for i in item[2]])
        A.extend([i[2] for i in item[2]])
        sim.extend(item[-1])
    V = np.asarray(V)
    A = np.asarray(A)
    sim = np.asarray( [ item if item>0 else 0.1 for item in sim])
    V_scores = sum(V*sim)/sum(sim)
    A_scores = sum(A*sim)/sum(sim)
    # print(V)
    # print(A)
    # print(sim)
    # print(V_scores,A_scores)
    return V_scores,A_scores


def print_array(arrays,count=0):
    """
        打印多维数组

    :param arrays:
    :param count:
    :return:
    """

    if type(arrays) not in [np.ndarray,list]:
        print(arrays,end=',')
        return

    print('[',end=' ')
    for items in arrays:
        print_array(items,count+1)

    if count==1:
        print(']')
    else:
        print(']',end=' ')



def evaluate(train_data,test_data,top_word=3,top_cluster=3,verbose=0):
    """
        对 模型 进行评价

    :param train_data: 训练集，已经聚完类
    :param test_data: 测试集
    :param top_word: 取 top n 个词进行算分
    :param top_cluster: 取 top n 个cluster 进行算 VA 值
    :param verbose: 数值越大，打印更多详情
    :return:
    """
    V_preds = []
    A_preds = []
    for word,V,A in test_data.values:
        # word=u'极好'
        V_pred,A_pred = predict(train_data,word,top_word=top_word,top_cluster=top_cluster,verbose=verbose)
        if verbose >0:
            print(word,V,A,V_pred,A_pred)
        V_preds.append(V_pred)
        A_preds.append(A_pred)
        # quit()
    V_mae = mean_absolute_error(test_data['Valence_Mean'].values, V_preds)
    A_mae = mean_absolute_error(test_data['Arousal_Mean'].values,A_preds)
    print('Valence Mean absolute error:%f'%V_mae)
    print('Arousal Mean absolute error:%f'%A_mae)

    from scipy.stats.stats import pearsonr
    V_p = pearsonr(test_data['Valence_Mean'].values,V_preds)[0]
    A_p = pearsonr(test_data['Arousal_Mean'].values,A_preds)[0]
    print('Valence Pearson correlation coefficient:%f'%V_p)
    print('Arousal Pearson correlation coefficient:%f'%A_p)
    return V_mae,A_mae,V_p,A_p


def cross_validation():
    for num_clusters in [1,5,20,50,100,150,200]:
        for top_word in [1,2,3,4,5]:
            for top_cluster in [1,2,3,4]:
                # num_clusters = 200
                # top_word = 3
                # top_cluster = 3
                print('='*40)
                print('num_clusters,top_word,top_cluster: %d,%d,%d:'%(num_clusters,top_word,top_cluster))
                count = 0
                V_mae_socre_list = []
                A_mae_socre_list = []
                V_p_socre_list = []
                A_p_socre_list = []
                for train_data, test_data in dutil.get_train_test_data(version='1'):
                    if count==0:
                        print('训练：')
                    else:
                        print('验证%d：'%count)
                    print('-'*40)
                    print('训练集和测试集个数分别为：%d,%d' % (len(train_data), len(test_data)))
                    train_data = train_data[['Word_jian', 'Valence_Mean', 'Arousal_Mean']]
                    test_data = test_data[['Word_jian', 'Valence_Mean', 'Arousal_Mean']]
                    # 聚类
                    knnCluster(train_data,num_clusters,verbose=0)
                    V_mae, A_mae, V_p, A_p = evaluate(train_data,test_data,top_word=top_word,top_cluster=top_cluster,verbose=0)
                    V_mae_socre_list += [V_mae]
                    A_mae_socre_list += [A_mae]
                    V_p_socre_list += [V_p]
                    A_p_socre_list += [A_p]
                    count+=1
                    print('-'*40)
                V_mae_socre_list += [np.mean(V_mae_socre_list[1:])]
                A_mae_socre_list += [np.mean(A_mae_socre_list[1:])]
                V_p_socre_list += [np.mean(V_p_socre_list[1:])]
                A_p_socre_list += [np.mean(A_p_socre_list[1:])]
                print(V_mae_socre_list)
                print(A_mae_socre_list)
                print(V_p_socre_list)
                print(A_p_socre_list)
                # quit()

def test1():
    # 测试模型
    train_data, test_data = dutil.get_train_test_data(version='1').next()
    print('训练集和测试集个数分别为：%d,%d'%(len(train_data),len(test_data)))
    # print test.head()
    # quit()
    train_data = train_data[['Word_jian', 'Valence_Mean', 'Arousal_Mean']]
    test_data = test_data[['Word_jian', 'Valence_Mean', 'Arousal_Mean']]
    # 聚类
    num_clusters = 5
    knnCluster(train_data,num_clusters,verbose=0)
    # quit()
    evaluate(train_data,test_data,top_word=3,top_cluster=3,verbose=0)

    # print(clusters[0])

if __name__ == '__main__':
    # test1()
    cross_validation()