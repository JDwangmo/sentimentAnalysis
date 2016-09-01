# coding=utf-8

import pandas as pd
import numpy as np

import networkx as nx
from collections import defaultdict
from itertools import permutations
from itertools import combinations
import gensim
import community as comg

threshold = 0  # 建边的阈值


dataname = '/home/jdwang/PycharmProjects/sentimentAnalysis/test2/cvaw1.csv'  # 训练集文件
vectormodel = '/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/50dim/wiki.zh.text.model'  # 模型文件

data = pd.read_csv(dataname,  # 读入训练集
                   sep=',',
                   encoding='gbk')
model = gensim.models.Word2Vec.load(vectormodel)  # 导入训练集


# 数据切分
def data_split(data, ratio=0.8):
    '''
    数据集分割：按ratio比例将数据集分割成训练集和测试集
    :param data: 全体数据
    :param ratio: 分割比例
    :return: 训练集和测试集
    '''
    np.random.seed(1)
    size = int(len(data) * ratio)  # 训练集长度
    shuffle = range(len(data))  # 训练集索引
    np.random.shuffle(shuffle)  # 随机打乱索引

    train = data.iloc[shuffle[:size]]
    test = data.iloc[shuffle[size:]]
    return train, test


train, test = data_split(data)


# 获取数据集的word：VA对
def data_dict(data):
    word_VA = {}
    data = data[['Word_jian', 'Valence_Mean', 'Arousal_Mean']].values

    for word, V, A in data:
        word_VA[word] = [V, A]
    return word_VA


word_VA = data_dict(train)


class Louvain(object):
    def __init__(self):
        self.MIN_VALUE = 0.0000001
        self.node_weights = {}

    @classmethod
    def convertIGraphToNxGraph(cls, igraph):
        node_names = igraph.vs["name"]
        edge_list = igraph.get_edgelist()
        weight_list = igraph.es["weight"]
        node_dict = defaultdict(str)

        for idx, node in enumerate(igraph.vs):
            node_dict[node.index] = node_names[idx]

        convert_list = []
        for idx in range(len(edge_list)):
            edge = edge_list[idx]
            new_edge = (node_dict[edge[0]], node_dict[edge[1]], weight_list[idx])
            convert_list.append(new_edge)

        convert_graph = nx.Graph()
        convert_graph.add_weighted_edges_from(convert_list)
        return convert_graph

    def updateNodeWeights(self, edge_weights):
        node_weights = defaultdict(float)
        for node in edge_weights.keys():
            node_weights[node] = sum([weight for weight in edge_weights[node].values()])
        return node_weights

    def getBestPartition(self, graph, param=1.):
        node2com, edge_weights = self._setNode2Com(graph)

        node2com = self._runFirstPhase(node2com, edge_weights, param)
        best_modularity = self.computeModularity(node2com, edge_weights, param)

        partition = node2com.copy()
        new_node2com, new_edge_weights = self._runSecondPhase(node2com, edge_weights)

        while True:
            new_node2com = self._runFirstPhase(new_node2com, new_edge_weights, param)
            modularity = self.computeModularity(new_node2com, new_edge_weights, param)
            if abs(best_modularity - modularity) < self.MIN_VALUE:
                break
            best_modularity = modularity
            partition = self._updatePartition(new_node2com, partition)
            _new_node2com, _new_edge_weights = self._runSecondPhase(new_node2com, new_edge_weights)
            new_node2com = _new_node2com
            new_edge_weights = _new_edge_weights
        return partition

    def computeModularity(self, node2com, edge_weights, param):
        q = 0
        all_edge_weights = sum(
            [weight for start in edge_weights.keys() for end, weight in edge_weights[start].items()]) / 2

        com2node = defaultdict(list)
        for node, com_id in node2com.items():
            com2node[com_id].append(node)

        for com_id, nodes in com2node.items():
            node_combinations = list(combinations(nodes, 2)) + [(node, node) for node in nodes]
            cluster_weight = sum([edge_weights[node_pair[0]][node_pair[1]] for node_pair in node_combinations])
            tot = self.getDegreeOfCluster(nodes, node2com, edge_weights)
            q += (cluster_weight / (2 * all_edge_weights)) - param * ((tot / (2 * all_edge_weights)) ** 2)
        return q

    def getDegreeOfCluster(self, nodes, node2com, edge_weights):
        weight = sum([sum(list(edge_weights[n].values())) for n in nodes])
        return weight

    def _updatePartition(self, new_node2com, partition):
        reverse_partition = defaultdict(list)
        for node, com_id in partition.items():
            reverse_partition[com_id].append(node)

        for old_com_id, new_com_id in new_node2com.items():
            for old_com in reverse_partition[old_com_id]:
                partition[old_com] = new_com_id
        return partition

    def _runFirstPhase(self, node2com, edge_weights, param):
        all_edge_weights = sum(
            [weight for start in edge_weights.keys() for end, weight in edge_weights[start].items()]) / 2
        self.node_weights = self.updateNodeWeights(edge_weights)
        status = True
        while status:
            statuses = []
            for node in node2com.keys():
                statuses = []
                com_id = node2com[node]
                neigh_nodes = [edge[0] for edge in self.getNeighborNodes(node, edge_weights)]

                max_delta = 0.
                max_com_id = com_id
                communities = {}
                for neigh_node in neigh_nodes:
                    node2com_copy = node2com.copy()
                    if node2com_copy[neigh_node] in communities:
                        continue
                    communities[node2com_copy[neigh_node]] = 1
                    node2com_copy[node] = node2com_copy[neigh_node]

                    delta_q = 2 * self.getNodeWeightInCluster(node, node2com_copy, edge_weights) - (self.getTotWeight(
                        node, node2com_copy, edge_weights) * self.node_weights[node] / all_edge_weights) * param
                    if delta_q > max_delta:
                        max_delta = delta_q
                        max_com_id = node2com_copy[neigh_node]

                node2com[node] = max_com_id
                statuses.append(com_id != max_com_id)

            if sum(statuses) == 0:
                break

        return node2com

    def _runSecondPhase(self, node2com, edge_weights):
        com2node = defaultdict(list)

        new_node2com = {}
        new_edge_weights = defaultdict(lambda: defaultdict(float))

        for node, com_id in node2com.items():
            com2node[com_id].append(node)
            if com_id not in new_node2com:
                new_node2com[com_id] = com_id

        nodes = list(node2com.keys())
        node_pairs = list(permutations(nodes, 2)) + [(node, node) for node in nodes]
        for edge in node_pairs:
            new_edge_weights[new_node2com[node2com[edge[0]]]][new_node2com[node2com[edge[1]]]] += edge_weights[edge[0]][
                edge[1]]
        return new_node2com, new_edge_weights

    def getTotWeight(self, node, node2com, edge_weights):
        nodes = [n for n, com_id in node2com.items() if com_id == node2com[node] and node != n]

        weight = 0.
        for n in nodes:
            weight += sum(list(edge_weights[n].values()))
        return weight

    def getNeighborNodes(self, node, edge_weights):
        if node not in edge_weights:
            return 0
        return edge_weights[node].items()

    def getNodeWeightInCluster(self, node, node2com, edge_weights):
        neigh_nodes = self.getNeighborNodes(node, edge_weights)
        node_com = node2com[node]
        weights = 0.
        for neigh_node in neigh_nodes:
            if node_com == node2com[neigh_node[0]]:
                weights += neigh_node[1]
        return weights

    def _setNode2Com(self, graph):
        node2com = {}
        edge_weights = defaultdict(lambda: defaultdict(float))
        for idx, node in enumerate(graph.nodes()):
            node2com[node] = idx
            for edge in graph[node].items():
                edge_weights[node][edge[0]] = edge[1]["weight"]
        return node2com, edge_weights


def makeSampleGraph():
    g = nx.Graph()

    data_word = data.Word_jian.values
    size = len(data_word)
    for i in xrange(size - 1):
        if(i%100 ==0):
            print '100个点完成'
        for j in xrange(i + 1, size):
            if (data_word[i] in model and data_word[j] in model):
                sim = model.similarity(data_word[i], data_word[j])
                if (sim > threshold):
                    g.add_edge(data_word[i], data_word[j], weight=sim)
    return g


# 计算unseen 词，移动到特定的社区cluster 后，只在与该unseen 连边的词中，按权汇总他们对unseen 词的贡献VA值
def calculate_VA(word, clusters, word_VA):
    weighted_VA = 0
    totle_sims = 0
    ok = False
    for word1 in clusters:
        if (word1 not in word_VA.keys()):
            continue
        sim = model.similarity(word, word1)
        if sim >= threshold:
            ok = True
            tmp = np.array(word_VA[word1])
            weighted_VA += tmp * sim
            totle_sims += sim
        ok = True
        tmp = np.array(word_VA[word1])
        weighted_VA += tmp * sim
        totle_sims += sim

    if (ok == False):  # 如果簇内没有可参考的VA词，返回[5,5]先，以后可以选择返回平均值。
        return [5, 5],False
    return weighted_VA / totle_sims,True


# 测试评估
def evaluation(test, result):
    '''
	预测评估
	:param test: 测试集真是结果
	:param result: 测试集预测结果
	:return:
	'''
    n = len(test)  # 测试集大小
    V_mae, V_r = 0, 0  # Valence的绝对均值误差和皮尔逊相关系数
    A_mae, A_r = 0, 0  # Arousal的绝对均值误差和皮尔逊相关系数
    V_A_mean, A_A_mean = np.mean(test, axis=0)  # Valence和Arousal的真实均值
    V_P_mean, A_P_mean = np.mean(result, axis=0)  # Valence和Arousal的预测均值
    V_std, A_std = np.std(test, axis=0)  # Valence和Arousal的真实方差
    V_P_std, A_P_std = np.std(result, axis=0)  # Valence和Arousal的测试均值

    print 'V_A_mean:', V_A_mean, '-------', 'V_P_mean:', V_P_mean
    print 'A_A_mean:', A_A_mean, '-------', 'A_P_mean:', A_P_mean
    print
    print 'V_std:', V_std, '-------', 'V_P_std:', V_P_std
    print 'A_std:', A_std, '-------', 'A_P_std:', A_P_std
    print
    print

    # 根据官方给出的方法计算测试指标
    for i in xrange(n):
        V_mae += np.abs(test[i, 0] - result[i, 0])
        A_mae += np.abs(test[i, 1] - result[i, 1])
        V_r += ((test[i, 0] - V_A_mean) / V_std) * ((result[i, 0] - V_P_mean) / V_P_std)
        A_r += ((test[i, 1] - A_A_mean) / A_std) * ((result[i, 1] - A_P_mean) / A_P_std)
    print 'Valence mean absolute error:', V_mae / n
    print 'Arousal mean absolute error:', A_mae / n
    print 'Valence pearson correlation coefficient:', V_r / (n - 1)
    print 'Arousal pearson correlation coefficient:', A_r / (n - 1)


def run():
    sample_graph = makeSampleGraph()
    #louvain = Louvain()
    #partition = louvain.getBestPartition(sample_graph)
    partition = comg.best_partition(sample_graph,resolution=.96)
    p = defaultdict(list)

    for node, com_id in partition.items():
        p[com_id].append(node)

    cnt = len(data)
    nodes2word = [[] for i in xrange(cnt)] #给出社区编号找里面有哪些单词
    num = 0
    for com, nodes in p.items():
        for n in nodes:
            print n,
            nodes2word[com].append(n)
            num += 1
        print ''

    print 'num',num
    word2nodes = {}
    for i in xrange(cnt):
        for word in nodes2word[i]:
            word2nodes[word] = i


    test_words = test[['No.', 'Word_jian', 'Valence_Mean', 'Arousal_Mean']].values
    test1 = test[['Valence_Mean', 'Arousal_Mean']].values
    size_test = len(test1)
    result = []
    none = 0
    for i in xrange(size_test):
        word = test_words[i, 1]
        if (word not in word2nodes.keys()):
            result.append([5, 5])
            print word, [5, 5]
            none += 1
            continue
        ans,falg = calculate_VA(word, nodes2word[word2nodes[word]], word_VA)
        result.append(ans)
        if(falg==False):
            none +=1
        print word, ans

    print '不在word2vec、或者没有簇内参考值、或者最高相似度无法达到阈值：', none
    result = np.array(result).round(1)  # 保留一位小数
    evaluation(test1, result)
    total = len(test1)
    v_inverse = 0
    a_inverse = 0
    mae_v = 0
    rmse_v = 0
    mae_a = 0
    rmse_a = 0
    for i in range(len(test)):
        if test_words[i, 2] > 5:
            if result[i, 0] < 5:
                v_inverse += 1
        else:
            if result[i, 0] > 5:
                v_inverse += 1
        if test_words[i, 3] > 5:
            if result[i, 1] < 5:
                a_inverse += 1
        else:
            if result[i, 1] > 5:
                a_inverse += 1
        mae_v += abs(test_words[i, 2] - result[i, 0])
        mae_a += abs(test_words[i, 3] - result[i, 1])
        rmse_v += (test_words[i, 2] - result[i, 0]) * (test_words[i, 2] - result[i, 0])
        rmse_a += (test_words[i, 3] - result[i, 1]) * (test_words[i, 3] - result[i, 1])
    mae_v = np.sqrt(mae_v / total)
    mae_a = np.sqrt(mae_a / total)
    rmse_v = np.sqrt(rmse_v / total)
    rmse_a = np.sqrt(rmse_a / total)
    print 'the mae of V :', mae_v, 'the mae of A :', mae_a
    print 'the rmse of V :', rmse_v, 'the rmse of A :', rmse_a
    print 'the InversePolarity of V is : ', v_inverse * 0.1 / total
    print 'the InversePolarity of V is : ', a_inverse * 0.1 / total
    df = pd.DataFrame({'No.': test_words[:, 0]})
    df['WORD'] = test_words[:, 1]
    df['TRUE_V'] = test_words[:, 2]
    df['PREDICT_V'] = result[:, 0]
    df['V_ERROR'] = test_words[:, 2] - result[:, 0]
    df['TRUE_A'] = test_words[:, 3]
    df['PREDICT_A'] = result[:, 1]
    df['A_ERROR'] = test_words[:, 3] - result[:, 1]
    # df = pd.DataFrame(columns=('No.', 'WORD', 'TRUE_V', 'PREDICT_V', 'TRUE_A', 'PREDICT_A'))
    # for i in xrange(size_t):
    # 	# 测试的词集，测试词集真实VA和预测VA
    # 	df.loc[i] =  test_words[i, 0], test_words[i, 1], test_words[i, 2], result[i][0], test_words[i, 3], result[i][1]
    # df['No.'].astype(int)
    df.to_csv('result_prodessing.csv',
              index=None,
              sep=',',
              encoding='utf-8')
    return


run()
