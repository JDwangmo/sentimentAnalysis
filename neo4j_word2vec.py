#coding=utf-8

import pandas as pd
import numpy as np
import gensim
import random
from py2neo import Graph, authenticate
from igraph import Graph as IGraph


authenticate('localhost:7474', 'neo4j', '1993526RUI')
graph = Graph("http://localhost:7474/db/data/")


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
				if sim > 0.5:
					V1, V2 = VA[i, 0], VA[j, 0]
					A1, A2 = VA[i, 1], VA[j, 1]

					write_dict = {'word1':w1, 'word2':w2,
								  'V1':V1, 'V2':V2,
								  'A1': A1, 'A2': A2,
								  'sim':sim}
					graph.cypher.execute(write_into_neo4j, write_dict)

# train, test = data_split(data)
# write_neo4j(train)


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


# 社区发现算法
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
		node['community'] = clusters.membership[idx]

	write_clusters_query = '''
	unwind {nodes} as n
	match (w:WORD)
	where w.name = n.name
	set w.community = toInt(n.community)
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

	clusters = [0]*len(all_clusters)
	if word in model:
		words = model.most_similar(word, topn=10)
		for w, sim in words:
			for item in all_clusters:
				if w in item.sim_words:
					clusters[item.cluster] += 1

		clusters = np.array(clusters)
		if clusters.sum() == 0:
			VA = train[['Valence_Mean', 'Arousal_Mean']].values
			VA = np.mean(VA, axis=0)
			return VA, False
		idx = np.argmax(clusters)
		sim_words = all_clusters[idx].sim_words
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
		with w.community as cluster, collect(w.name) as sim_words
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

	for i in xrange(size_t):
		word = test_words[i, 0]
		tmp, flag = predict(word, train, all_clusters)
		if flag == None:
			none += 1
		result[i] = tmp

	evaluation(test_VA, result)
	for i in xrange(size_t):
		# 测试的词集，测试词集真实VA和预测VA
		print test_words[i, 0], test_words[i, 1:], result[i]

run()