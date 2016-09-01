#coding=utf-8

import pandas as pd
import numpy as np
import gensim
import random
from py2neo import Graph, authenticate
from igraph import Graph as IGraph


authenticate('localhost:7474', 'neo4j', '123456') #设置身份验证参数
graph = Graph("http://localhost:7474/db/data/") #连接到验证图形数据库
threshold = 0.4

dataname = 'cvaw1.csv'					# 训练集文件
vectormodel = 'wiki.zh.text.model'		# 模型文件
data = pd.read_csv(dataname,			# 读入训练集
				   sep=',',
				   encoding='gbk')
model = gensim.models.Word2Vec.load(vectormodel)	# 导入训练集


# VA的散点图
def scatter_show():
	import matplotlib.pyplot as plt
	VA = data[['Valence_Mean', 'Arousal_Mean']].values
	V, A = VA[:, 0], VA[:, 1]
	plt.xlim([1, 9]), plt.ylim([1, 9])
	plt.plot(V, A, 'or')
	plt.show()


# 数据切分
def data_split(data, ratio=0.8):
	'''
	数据集分割：按ratio比例将数据集分割成训练集和测试集
	:param data: 全体数据
	:param ratio: 分割比例
	:return: 训练集和测试集
	'''
	np.random.seed(1)
	size = int(len(data)*ratio)		# 训练集长度
	shuffle = range(len(data))		# 训练集索引
	np.random.shuffle(shuffle)		# 随机打乱索引

	train = data.iloc[shuffle[:size]]
	test = data.iloc[shuffle[size:]]
	return train, test


# 写入neo4j数据库
def write_neo4j(data):

	'''
	将词和词对应的VA写入neo4j，词与词之间的权重为其欧氏距离（距离大于5才记录）
	:param data: 待写入数据
	:return:
	'''
	write_into_neo4j = '''
		merge (w1: WORD{name:{word1}, VALENCE:{V1}, AROUSAL:{A1}} )
		merge (w2: WORD{name:{word2}, VALENCE:{V2}, AROUSAL:{A2}} )
		create unique (w1)-[w:SIM]-(w2)
		set w.WEIGHT = {sim}
		'''
	words = data.Word_jian.values
	VA = data[['Valence_Mean', 'Arousal_Mean']].values
	size = len(VA)
	for i in range(size - 1):
		if i%100 == 0:
			print '完成一百个点'
		for j in range(i + 1, size):

			w1, w2 = words[i], words[j]
			if w1 in model and w2 in model:
				sim = model.similarity(w1, w2)
				if sim > threshold:
					V1, V2 = VA[i, 0], VA[j, 0]
					A1, A2 = VA[i, 1], VA[j, 1]

					write_dict = {'word1':w1, 'word2':w2,
								  'V1':V1, 'V2':V2,
								  'A1': A1, 'A2': A2,
								  'sim':sim}
					graph.cypher.execute(write_into_neo4j, write_dict)

train, test = data_split(data)
write_neo4j(data) #全体一起跑！


# pagerank算法
def pagerank():
	query = '''
	match (w1:WORD)-[r:SIM]-(w2:WORD)
	return w1.name, w2.name, r.WEIGHT
	'''
	ig = IGraph.TupleList(graph.cypher.execute(query), weights=True)
	pg = ig.pagerank()
	pgvs = []
	for p in zip(ig.vs, pg):
		pgvs.append( {'name':p[0]['name'], 'pg':p[1]} )

	write_clusters_query = '''
	unwind {nodes} as n
	match (w:WORD)
	where w.name = n.name
	set w.pagerank = n.pg
	'''
	graph.cypher.execute(write_clusters_query, nodes = pgvs)
# pagerank()


# 基于随机游走的社区发现算法
def community_detection():
	query = '''
	match (w1:WORD)-[r:SIM]-(w2:WORD)
	return w1.name, w2.name, r.WEIGHT
	'''
	ig = IGraph.TupleList(graph.cypher.execute(query), weights=True)
	clusters = IGraph.community_walktrap(ig, weights='weight').as_clustering()

	nodes = [ {'name':node['name']} for node in ig.vs]
	for node in nodes:
		idx = ig.vs.find( name=node['name']).index
		node['community_weighted_graph'] = clusters.membership[idx]

	write_clusters_query = '''
	unwind {nodes} as n
	match (w:WORD)
	where w.name = n.name
	set w.community_weighted_graph = toInt(n.community_weighted_graph)
	'''
	graph.cypher.execute(write_clusters_query, nodes=nodes)
community_detection()


# 获取数据集的word：VA对
def data_dict(data):

	word_VA = {}
	data = data[['Word_jian','Valence_Mean', 'Arousal_Mean']].values

	for word, V, A in data:
		word_VA[word] = [V, A]
	return word_VA


# 相似词查询
def similarity_not_word2vec(word, word_VA):
	'''
	在训练集中找相近的词，此函数用来处理如下情况：
	当word没有出现在word2vec预训练词中时，解决方案如下：
	计算训练集中的每个词与word的相似度，取相似度最高的钱5个
	:param word: 待查找的词
	:param word_VA:	训练集全体数据集合
	:return: 相似度最高的前5个词
	'''
	words = word_VA.keys()			# 训练集词集合
	size = len(words)
	sims = {}						# 存储相似词及相似度
	for i in xrange(size):
		if words[i] in model:		# 判断词是否在word2vec预训练词中
			sim = model.similarity(word, words[i])		# 计算word与训练集中的词的相似度
			sims[words[i]] = sim						# 保存与每个词的相似度
	if len(sims) == 0:									# 如果训练集中的词全都不在word2vec预训练词集中，则随机返回训练集中的一个词
		return random.choice(list(word_VA.iteritems()))
	sorted_sims = sorted(sims.iteritems(), key=lambda x:x[1], reverse=True)
	return sorted_sims[:5]


# 预测
def predict(word, train, all_clusters):

	word_VA = data_dict(train)

	clusters = [0]*len(all_clusters) #全列表设置为0
	if word in model:
		words = model.most_similar(word, topn=10) #取出前10个相似词。未来会使用similarity_not_word2vec 函数代替，提高识别数目！！！！（改进1）
		for w, sim in words:
			for item in all_clusters: #所有的社区中，取出一个社区
				if w in item.sim_words: #item.sim_words 代表社区的各个节点
					clusters[item.cluster] += 1 #item.cluster 代表该社区的编号

		clusters = np.array(clusters)
		if clusters.sum() == 0: #如果相似的词没有在一个社区中出现，返回所有训练集的平均值。
			VA = train[['Valence_Mean', 'Arousal_Mean']].values
			VA = np.mean(VA, axis=0)
			return VA, False
		idx = np.argmax(clusters) #没有考虑到，如果全部相似词都没有在一个社区中。这时候应该是设置为全体训练集的均值！！！此处改进与改进1是紧密联系的。（改进2）
		sim_words = all_clusters[idx].sim_words #取出该社区的所有节点，其实令结果等于该社区的平均值。这个方法不好，应该选择该社区中与unseed 单词相似的词，并且相似程度作为权值，最终才定下VA值！！（改进3）
		VA = []
		for w, sim in words:
			if w in sim_words:
				VA.append(word_VA[w])
		VA = np.array(VA)
		result = VA.mean(axis=0)
		return result, True
	else:
		VA = train[['Valence_Mean', 'Arousal_Mean']].values 
		VA = np.mean(VA, axis=0)
		return VA, False

def predict2(word, word_VA, mean, community):
	'''
    :param word:unseen词
    :param word_VA: 训练词对应的VA值
    :param clusters: unseen所在的簇
    :return:
    '''



	ok = False
	weighted_VA, totle_sims = 0, 0
	for word2 in community:  # 能在同一个簇中就证明能够加边，也就证明了在model中。
		if (word2 not in model or word2 not in word_VA.keys()):  # 如果不在模型中、或者不在训练词中。得不到VA参考值。
			continue
		sim = model.similarity(word, word2)
		if (sim < threshold):
			continue
		print word2,sim,
		ok = True
		tmp = np.array(word_VA[word2])
		weighted_VA += tmp * sim
		totle_sims += sim
	if (ok == False):
		print []
		print 'VA_ans: ', mean
		print ''
		return mean, False  # 如果unseen因为不在模型中，或者训练词不足以预测出该词，就返回全体训练词的均值。
	result = weighted_VA / totle_sims
	print 'VA_ans: ', result
	print ''
	return result, True

# 测试评估
def evaluation(test, result):
	'''
	预测评估
	:param test: 测试集真是结果
	:param result: 测试集预测结果
	:return:
	'''
	n = len(test)									# 测试集大小
	V_mae, V_r = 0, 0								# Valence的绝对均值误差和皮尔逊相关系数
	A_mae, A_r = 0, 0								# Arousal的绝对均值误差和皮尔逊相关系数
	V_A_mean, A_A_mean = np.mean(test, axis=0)		# Valence和Arousal的真实均值
	V_P_mean, A_P_mean = np.mean(result, axis=0)	# Valence和Arousal的预测均值
	V_std, A_std = np.std(test, axis=0)				# Valence和Arousal的真实方差
	V_P_std, A_P_std = np.std(result, axis=0)		# Valence和Arousal的测试均值

	print 'V_A_mean:', V_A_mean, '-------', 'V_P_mean:', V_P_mean
	print 'A_A_mean:', A_A_mean, '-------', 'A_P_mean:', A_P_mean
	print
	print 'V_std:', V_std, '-------', 'V_P_std:', V_P_std
	print 'A_std:', A_std, '-------', 'A_P_std:', A_P_std
	print
	print

	# 根据官方给出的方法计算测试指标
	for i in xrange(n):
		V_mae += np.abs(test[i, 0]-result[i, 0])
		A_mae += np.abs(test[i, 1]-result[i, 1])
		V_r += ((test[i, 0]-V_A_mean)/V_std) * ((result[i, 0]-V_P_mean)/V_P_std)
		A_r += ((test[i, 1]-A_A_mean)/A_std) * ((result[i, 1]-A_P_mean)/A_P_std)
	print 'Valence mean absolute error:', V_mae/n
	print 'Arousal mean absolute error:', A_mae/n
	print 'Valence pearson correlation coefficient:', V_r/(n-1)
	print 'Arousal pearson correlation coefficient:', A_r/(n-1)


def run():
	sim_words_query = '''
		match (w:WORD)
		with w.community_weighted_graph as cluster, collect(w.name) as sim_words
		return cluster, sim_words
		order by cluster
		'''

	all_clusters = graph.cypher.execute(sim_words_query)

	train, test = data_split(data)
	test_words = test[['Word_jian', 'Valence_Mean', 'Arousal_Mean']].values
	test_VA = test[['Valence_Mean', 'Arousal_Mean']].values
	size_t = len(test)
	none = 0
	result = np.zeros((size_t, 2))
	VA = train[['Valence_Mean', 'Arousal_Mean']].values
	mean = np.mean(VA, axis=0)
	word_VA = data_dict(train)

	word2clusters = {}

	for num,item in enumerate(all_clusters):  # 所有的社区中，取出一个社区
		for word in item.sim_words:  # item.sim_words 代表社区的各个节点
			word2clusters[word] = num  # item.cluster 代表该社区的编号

	for i in xrange(size_t):
		word = test_words[i, 0]
		if(word not in word2clusters.keys()): # 不在一个簇中。
			result[i] = mean
			none += 1
			print word
			print '[]'
			print 'VA_ans: ', result[i]
			print ''
			continue
		community = all_clusters[word2clusters[word]]
		tmp, flag = predict2(word, word_VA, mean, community)
		if flag == False:
			none += 1
		result[i] = tmp
	print 'none', none

	evaluation(test_VA, result)

run()