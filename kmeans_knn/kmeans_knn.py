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
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from dataset.data_util import DataUtil

dutil = DataUtil()


def showChart(points, position):  # 绘制图表
    plt.scatter(points[:, 0], points[:, 1], c=position)  # scatter绘制散点
    plt.show()
    return


def knnCluster(train,
               num_clusters=5,
               columns=None,
               verbose=0):
    """
        Kmeans 聚类

    :param train: 训练集
    :param num_clusters: 聚类个数
    :param columns: 指定使用哪一列聚类
    :param verbose: 0,1,2
    :return:
    """
    # words = train[['Word_jian']].values
    if columns is None:
        columns = ['Valence_Mean', 'Arousal_Mean']
    print('columns:%s' % columns)
    print('-' * 40)
    filter_xy = train[columns].values
    random_seed = 1
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_seed)  # n_clusters 聚类个数
    class_arr = kmeans.fit_predict(filter_xy)
    if verbose > 1:
        showChart(train[['Valence_Mean', 'Arousal_Mean']].values, class_arr)

    # clusters = [[] for i in range(num_clusters)]
    # for i in range(len(class_arr)):
    #     clusters[class_arr[i]].append(words[i])
    # 肘部法则，计算loss
    loss = sum(np.min(cdist(filter_xy, kmeans.cluster_centers_, 'euclidean'), axis=1)) / len(filter_xy)
    # print(loss)
    # 将聚类结果保存
    train.loc[:, 'Cluster_ID'] = class_arr
    # return clusters
    return loss


def predict(train_data, word, top_word=3, top_cluster=3,
            threshold=0.3,
            use_emotional_lexicon=True,
            verbose=0):
    """
        km+knn预测

    :param train_data:
    :param word:
    :param top_word:
    :param top_cluster:
    :param threshold:
    :param use_emotional_lexicon: 使用情感词库
    :param verbose:
    :return:
    """
    cluster_scores = []
    # print(train_data.head())
    for cluster_id, group in train_data.groupby(by=['Cluster_ID']):
        # print('cluster %d: %d samples'%(cluster_id,len(group)))
        # 计算相似性
        words_sim = np.asarray([dutil.get_word_similarity(word, item) for item in group['Word_jian'].values])
        # 获取相似性降序的索引
        sorted_index = np.argsort(words_sim)[-1::-1]
        # print(sorted_index)

        top_words = group[['Word_jian', 'Valence_Mean', 'Arousal_Mean']].values[sorted_index[:top_word]]
        top_words_sim = words_sim[sorted_index[:top_word]]
        score = np.mean(words_sim[sorted_index[:top_word]])
        # print(group.iloc[sorted_index])
        # print(top_words_sim)
        # [print(item) for item in top_words]
        # print(score)
        cluster_scores.append([cluster_id, score, top_words, top_words_sim])
    # print(cluster_scores)
    # 根据分数升序排序
    sorted_cluster_scores = sorted(cluster_scores, key=lambda x: x[1], reverse=True)
    if verbose > 1:
        print_array(sorted_cluster_scores)

    V, A = get_predict_result(word, sorted_cluster_scores, top=top_cluster,
                              threshold=threshold,
                              use_emotional_lexicon=use_emotional_lexicon,
                              verbose=verbose,
                              )

    return V, A


def predict1(train_data, word, top_word=3,
             use_kmeans=True,
            use_emotional_lexicon=True,
             revise = True,
            verbose=0):
    """
        topK(w2v) + KM + 情感词处理 + 最后修正

    :param train_data:
    :param word:
    :param top_word:
    :param top_cluster:
    :param threshold:
    :param use_emotional_lexicon: 使用情感词库
    :param verbose:
    :return:
    """
    # cluster_scores = []
    # print(train_data.head())
    # print('cluster %d: %d samples'%(cluster_id,len(group)))
    # 计算相似性
    words_sim = np.asarray([dutil.get_word_similarity(word, item) for item in train_data['Word_jian'].values])
    # 获取相似性降序的索引
    sorted_index = np.argsort(words_sim)[-1::-1]
    # print(sorted_index)

    top_words = train_data[['Cluster_ID','Word_jian', 'Valence_Mean', 'Arousal_Mean']].values[sorted_index[:top_word]].tolist()
    top_words_sim = words_sim[sorted_index[:top_word]]
    # score = np.mean(words_sim[sorted_index[:top_word]])
    # print(train_data.iloc[sorted_index])
    # print(top_words_sim)
    # [print(item) for item in top_words]
    # print(score)
    topn_result = [item+[sim] for item,sim in zip(top_words, top_words_sim)]
    # topn_result = sorted(topn_result, key=lambda x: x[1], reverse=True)
    if verbose > 1:
        print_array(topn_result)
    if use_kmeans:
        # 根据分数升序排序
        cluster_score = dict()
        for items in topn_result:
            cluster_id, w,V,A,sim = items
            if cluster_score.has_key(cluster_id):
                cluster_score[cluster_id] += sim
            else:
                cluster_score[cluster_id] = sim
        cluster_id_list=cluster_score.keys()

        if len(set(cluster_id_list)) > 1:
            selected_cluster_id = np.asarray(cluster_id_list)[np.argsort(cluster_score.values())[1:]]
        else:
            selected_cluster_id = list(set(cluster_id_list))
        # print(selected_cluster_id)
        if verbose > 0:
            print('去除掉：')
            print_array([items for items in topn_result if items[0] not in selected_cluster_id])

        topn_result = [items for items in topn_result if items[0] in selected_cluster_id]
    # print_array(topn_result)
    # quit()
    # print(TV,TA)
    V = [items[2] for items in topn_result]
    A = [items[3] for items in topn_result]
    sim = [items[-1] for items in topn_result]
    sim = np.asarray([item if item > 0 else 0.1 for item in sim])
    if use_emotional_lexicon and not all(np.asarray(V) > 5) and not all(np.asarray(V) < 5):
        # 当出现冲突时，进行修正
        # 修正前
        V_scores = sum(np.asarray(V) * sim) / sum(sim)

        # 修正后
        TV, TA = dutil.get_emotional_lexicon_info(word,version=1)
        if verbose > 0:
            print('原本分数：%f, 进行修正, TV:%d,TA:%d' % (V_scores, TV, TA))

        if TV == 1:
            V = [V[index] for index in range(len(V)) if V[index] >= 5]
            sim = [sim[index] for index in range(len(V)) if V[index] >= 5]
            # print(V)
        elif TV == -1:
            # TV==-1
            V = [V[index] for index in range(len(V)) if V[index] <= 5]
            sim = [sim[index] for index in range(len(V)) if V[index] <= 5]
        V = np.asarray(V)
        V_scores = sum(V * sim) / sum(sim)
        if revise:
            if TV==-1 and V_scores>5 or TV==1 and V_scores<5:
                if verbose>0:
                    print('修正：%f--》%f'%(V_scores,10-V_scores))
                V_scores =10-V_scores

    else:
        V = np.asarray(V)
        A = np.asarray(A)
        V_scores = sum(V * sim) / sum(sim)
        A_scores = sum(A * sim) / sum(sim)
    # print(V)
    # print(V_scores)
    # quit()

    return V_scores, A_scores




def get_predict_result(word,
                       scores,
                       top=1,
                       threshold=0.,
                       use_emotional_lexicon =True,
                       verbose=0,
                       ):
    """
        根据近似性结果，去除 top-n 聚类之外的词，进行算 VA值，
        最多取 top 个，最少取1个，如果 top 个内，近似值 大于 某个阈值才计入

    :param word: 待预测词
    :param scores: 得分列表
    :param top: top 个聚类
    :param threshold: 阈值
    :param use_emotional_lexicon: 是否使用情感词汇修正结果
    :return:V_scores, A_scores
    """
    scores = scores[:top]
    V = []
    A = []
    sim = []
    for index, item in enumerate(scores):
        if index < 1 or item[3][0] > threshold:
            V.extend([i[1] for i in item[2]])
            A.extend([i[2] for i in item[2]])
            sim.extend(item[-1])

    A = np.asarray(A)
    sim = np.asarray([item if item > 0 else 0.1 for item in sim])
    A_scores = sum(A * sim) / sum(sim)
    if use_emotional_lexicon:
        if not all(np.asarray(V)>5) and not all(np.asarray(V)<5):
            # 当出现冲突时，进行修正
            # 修正前
            V_scores = sum(np.asarray(V) * sim) / sum(sim)

            # 修正后
            TV,TA = dutil.get_emotional_lexicon_info(word,version=1)
            if verbose>0:
                print('原本分数：%f, 进行修正, TV:%d,TA:%d'%(V_scores,TV,TA))

            if TV==1:
                V = [V[index] for index in range(len(V)) if V[index]>=5]
                sim = [sim[index] for index in range(len(V)) if V[index]>=5]
                # print(V)
            elif TV==-1:
                # TV==-1
                V = [V[index] for index in range(len(V)) if V[index] <= 5]
                sim = [sim[index] for index in range(len(V)) if V[index] <= 5]
    V = np.asarray(V)
    V_scores = sum(V * sim) / sum(sim)
    # print(V)
    # print(A)
    # print(sim)
    # print(V_scores,A_scores)
    return V_scores, A_scores


def print_array(arrays, count=0):
    """
        打印多维数组

    :param arrays:
    :param count:
    :return:
    """

    if type(arrays) not in [np.ndarray, list,tuple]:
        print(arrays, end=',')
        return

    print('[', end=' ')
    for items in arrays:
        print_array(items, count + 1)

    if count == 1:
        print(']')
    else:
        print(']', end=' ')


def evaluate(train_data, test_data,
             predict_method = 0,
             top_word=3,
             top_cluster=3,
             threshold=0.3,
             use_kmeans=True,
             revise=True,
             use_emotional_lexicon=True,
             verbose=0):
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
    for No, word, V, A in test_data.values:
        # word=u'敏锐'
        # print(No, word, V, A)
        if predict_method == 0:
            V_pred, A_pred = predict(train_data, word, top_word=top_word, top_cluster=top_cluster,
                                     threshold=threshold,
                                     use_emotional_lexicon=use_emotional_lexicon,
                                     verbose=verbose)
        elif predict_method==1:
            V_pred, A_pred = predict1(train_data, word, top_word=top_word,
                                      use_kmeans=use_kmeans,
                                      revise=revise,
                                      use_emotional_lexicon=use_emotional_lexicon,
                                      verbose=verbose)
        if verbose > 0:
            print('%d,%s,%f,%f,%f,%f,%f,%f'%(No, word, V, A, V_pred, A_pred,abs(V_pred-V),abs(A_pred-A)))
        V_preds.append(V_pred)
        A_preds.append(A_pred)
        # quit()

    test_data.loc[:, 'V_pred'] = V_preds
    test_data.loc[:, 'A_pred'] = A_preds

    V_mae = mean_absolute_error(test_data['Valence_Mean'].values, V_preds)
    A_mae = mean_absolute_error(test_data['Arousal_Mean'].values, A_preds)
    print('Valence Mean absolute error:%f' % V_mae)
    print('Arousal Mean absolute error:%f' % A_mae)

    from scipy.stats.stats import pearsonr
    V_p = pearsonr(test_data['Valence_Mean'].values, V_preds)[0]
    A_p = pearsonr(test_data['Arousal_Mean'].values, A_preds)[0]
    print('Valence Pearson correlation coefficient:%f' % V_p)
    print('Arousal Pearson correlation coefficient:%f' % A_p)
    return V_mae, A_mae, V_p, A_p


def select_kmeans_k():
    """
        # 通过肘部法则，选出合适 的k

    :return:
    """

    train_data, test_data = dutil.get_train_test_data(version='1').next()
    print('训练集和测试集个数分别为：%d,%d' % (len(train_data), len(test_data)))
    # quit()
    train_data = train_data[['Word_jian', 'Valence_Mean', 'Arousal_Mean']]
    columns = ['Valence_Mean']
    # 聚类
    num_clusters_list = xrange(1, 100)
    loss_list = []
    for num_clusters in num_clusters_list:
        loss = knnCluster(train_data,
                          num_clusters,
                          columns=columns,
                          verbose=0)
        loss_list.append(loss)
    xticks = range(min(num_clusters_list), max(num_clusters_list) + 1, 1)
    yticks = np.arange(min(loss_list) - 0.1, max(loss_list) + 0.1, 0.1)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.axis([num_clusters_list[0] - 1, num_clusters_list[-1] + 1, min(loss_list) - 0.1, max(loss_list) + 0.1])
    plt.plot(num_clusters_list, loss_list, 'b*-')
    plt.title(','.join(columns), fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.xlabel('num_clusters', fontsize=18)
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.clf()


def cross_validation():
    for num_clusters in [8]:
        for top_word in [ 8,20]:
            for top_cluster in [ 4]:
                # num_clusters = 200
                # top_word = 3
                # top_cluster = 3
                print('=' * 40)
                print('num_clusters,top_word,top_cluster: %d,%d,%d:' % (num_clusters, top_word, top_cluster))
                count = 0
                V_mae_socre_list = []
                A_mae_socre_list = []
                V_p_socre_list = []
                A_p_socre_list = []
                for train_data, test_data in dutil.get_train_test_data(version='2'):
                    if count == 0:
                        print('训练：')
                    else:
                        print('验证%d：' % count)
                    print('-' * 40)
                    print('训练集和测试集个数分别为：%d,%d' % (len(train_data), len(test_data)))
                    train_data = train_data[['Word_jian', 'Valence_Mean', 'Arousal_Mean']]
                    test_data = test_data[['No.', 'Word_jian', 'Valence_Mean', 'Arousal_Mean']]
                    # 聚类
                    knnCluster(train_data,
                               num_clusters,
                               columns=['Valence_Mean'],
                               verbose=0)
                    V_mae, A_mae, V_p, A_p = evaluate(
                        train_data, test_data, top_word=top_word, top_cluster=top_cluster,
                        predict_method=1,
                        threshold=0.3,
                        use_kmeans=False,
                        revise=False,
                        use_emotional_lexicon=False,
                        verbose=0)
                    V_mae_socre_list += [V_mae]
                    A_mae_socre_list += [A_mae]
                    V_p_socre_list += [V_p]
                    A_p_socre_list += [A_p]
                    count += 1
                    print('-' * 40)
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
    print('训练集和测试集个数分别为：%d,%d' % (len(train_data), len(test_data)))
    print('=' * 40)
    # print(test_data.head())
    # print(train_data.shape)
    # quit()
    train_data = train_data[['Word_jian', 'Valence_Mean', 'Arousal_Mean']]

    test_data = test_data[['No.', 'Word_jian', 'Valence_Mean', 'Arousal_Mean']]
    # 聚类
    num_clusters = 12
    knnCluster(train_data,
               num_clusters,
               columns=['Arousal_Mean'],
               verbose=0)

    evaluate(train_data,
             test_data,
             predict_method=1,
             top_word=20,
             top_cluster=3,
             threshold=0.3,
             use_kmeans=True,
             use_emotional_lexicon=False,
             revise=False,
             verbose=2,
             )
    # print(test_data.head())
    dutil.save_data_to_csv(test_data,
                           '/home/jdwang/PycharmProjects/sentimentAnalysis/kmeans_knn/kmeans_knn_final_test_result.csv')
    # print(clusters[0])


if __name__ == '__main__':
    # select_kmeans_k()
    # test1()
    cross_validation()
