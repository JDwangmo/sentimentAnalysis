# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-28'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function

choice = 3
count =0
with open('/home/jdwang/PycharmProjects/sentimentAnalysis/kmeans_knn/cross_validation/knn_社区_最后修正_5fold_cv.txt','r') as fin:
    for line in fin:
        line = line.strip()
        if choice ==1:
            if line.startswith('num_clusters,top_word,top_cluster: '):
                line = line.replace('num_clusters,top_word,top_cluster: ','').replace(':','')
                print(line)
        elif choice==2:
            if line.startswith('['):
                line = line.replace('[','').replace(']','')
                if count%4==2:
                    print(line)
                count+=1
        elif choice==3:
            if line.startswith('Valence mean absolute error:'):
                line = line.replace('Valence mean absolute error: ', '')
                print(line)
        elif choice==4:
            if line.startswith('Valence pearson correlation coefficient: '):
                line = line.replace('Valence pearson correlation coefficient: ', '')
                print(line)
