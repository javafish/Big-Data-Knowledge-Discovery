import xml.etree.cElementTree as ET
import csv
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib as plt
from gensim.models import doc2vec
import logging
import numpy as np
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn import mixture
from itertools import combinations
from sklearn.cluster import Birch

def preprocess():
    """
    :return:
    """
    temp = ""                            #每一篇论文的联合作者拼成以，相隔的字符串
    temp1 = ""                           #论文作者名字内部没有空格，名字与名字间以空格隔开，用于doc2vec
    temp2 = ""                           #暂存每篇论文实际属于哪个Wei Wang
    str1 = ""                            #用于去掉空格，因为doc2vec切词默认按照空格切词
    rid = []                             #记录的ID
    author = []                          #作者
    author_doc = []                      #作者，为了适应doc2vec的格式
    title = []                           #题目
    year = []                            #发表年份
    belong = []                          #属于哪一个Wei Wang写的
    booktitle = []                       #发表的期刊或会议
    #把xml文件读进一课树
    tree = ET.ElementTree(file="dblp.xml")
    #生成1—435条记录的ID
    for i in range(1, 436):
        rid.append(i)
    #读取xml树中的内容
    for element in tree.iter():
        if element.tag == "author" or element.tag == "editor":
            if "Wei Wang" in element.text.strip():
                temp2 = element.text.strip()
                temp = temp + "Wei Wang" + ","
            else:
                temp = temp + element.text.strip() + ","
            str1 = "".join(element.text.strip().split())
            temp1 = temp1 + str1 + " "

        if element.tag == "title":
            author.append(temp.strip(","))
            author_doc.append(temp1.strip())
            belong.append(temp2)
            temp = ""
            temp1 = ""
            title.append(element.text.strip())
        if element.tag == "year":
            year.append(element.text.strip())
        if element.tag == "journal" or element.tag == "booktitle":
            booktitle.append(element.text.strip())
    university = []
    pattern = {'Wei Wang 0001': 'University of Waterloo', 'Wei Wang 0002': 'Nanjing University',
               'Wei Wang 0003': 'State University of New York at Albany',
               'Wei Wang 0004': 'Fudan University', 'Wei Wang 0005': 'Zhejiang University',
               'Wei Wang 0006': 'Language Weaver', 'Wei Wang 0007': 'Chinese Academy of Sciences',
               'Wei Wang 0008': 'MIT', 'Wei Wang 0009': 'Fudan University',
               'Wei Wang 0010': 'University of California Los Angeles'}
    university = [pattern[x] if x in pattern else x for x in belong]
    d = {"id":rid,"author": author, "belong": belong, "university": university, "title": title, "booktitle": booktitle,
         "year": year}
    training_df = pd.DataFrame(data=d)
    training_df.to_csv("dblp.csv",header=True,index=False)
    print("This data is stored in the Dataframe")

def get_df(file_name):
    training_df=pd.read_csv("dblp.csv",encoding='gbk')
    return training_df
def to_txt(num_features):
    with open("dblp.csv") as cfile:
        reader = csv.DictReader(cfile)
        f = open("dblp.txt", "w", encoding="utf-8")
        for row in reader:
            if num_features==1:
                f.write(row['author']+"\n")
            elif num_features==2:
                f.write(row['author'] + " " + row['university']+ "\n")
            elif num_features==3:
                f.write(row['author'] + " "+ row['university'] +" "+row['booktitle'] +"\n")
            elif num_features==4:
                f.write(row['author'] + " " + row['university'] + " " + row['title'] + " " + row['booktitle'] + "\n")
            else:
                print("Intput error!")
def to_doc():
    with open("dblp.csv") as cfile:
        reader = csv.DictReader(cfile)
        f = open("doc.txt", "w", encoding="utf-8")
        for row in reader:
            f.write("".join(row['author'].split())+" "+"".join(row['university'].split())+"\n")

def load_dataset(file_name):
    file = codecs.open("dblp.txt", 'r', encoding=u'utf-8', errors='ignore')
    dataset = []
    for line in file:
        dataset.append(line.strip())
    file.close()
    return dataset

#def bunch_tfidf(dataset):


def kmeans_tfidf(dataset,minibatch):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000, min_df=2, use_idf=True,stop_words='english')  # 计算TF-IDF
    X = vectorizer.fit_transform(dataset)  # 特征处理类
    # 使用采样数据还是原始数据训练k-means，
    if minibatch:
        km = MiniBatchKMeans(n_clusters=10, init='k-means++', n_init=1, init_size=1000, batch_size=1000,
                             verbose=False)
    else:
        km = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=1, verbose=False, random_state=0)  # 随机数种子
    km.fit(cosine_similarity(X))

    result = list(km.predict(cosine_similarity(X)))
    return result

def kmeans_doc2vec(file_name):
    sentences = doc2vec.TaggedLineDocument(file_name)
    model = doc2vec.Doc2Vec(sentences,  # 语料集
                            size=40,  # 是指特征向量的维度
                            window=3,  # 表示当前词与预测词在一个句子中的最大距离是多少
                            )
    model.save_word2vec_format("doc2vec_result.txt")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    num_clusters = 10
    km = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=1, verbose=False, random_state=0)
    result_doc2vec = list(km.fit_predict(model.docvecs))
    return result_doc2vec

def gmm_tfidf(dataset):
    #dataset=load_dataset(file_name)
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000, min_df=2, use_idf=True,
                                 stop_words='english')  # 计算TF-IDF
    X = vectorizer.fit_transform(dataset)  # 特征处理类
    gmm = mixture.GMM(n_components=10, n_iter=5000,covariance_type='full')
    result_gmm = list(gmm.fit_predict(X.toarray()))
    return result_gmm

#层次聚类
def birch_tfidf(dataset):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000, min_df=2, use_idf=True,
                                 stop_words='english')  # 计算TF-IDF
    X = vectorizer.fit_transform(dataset)  # 特征处理类
    result_birch = Birch(n_clusters=10,threshold=0.5, branching_factor=50).fit_predict(X)
    return result_birch

def get_result(result):
    training_df=pd.read_csv("dblp.csv",encoding='gbk')
    result_df=pd.DataFrame(result,columns=["result"])
    pre_df=training_df.join(result_df)
    pre_df.drop(['author', 'booktitle', 'title', 'year','university'], axis=1, inplace=True)
    for i in range(1,11):
        cluster_name = pre_df[pre_df['result'].isin([int(i - 1)])].drop(['result'], axis=1)
        cluster_name.to_csv("cluster_" + str(i) + ".csv", header=True, index=False)
    return pre_df

def precision_rate(training_df,pre_df):
    total_list = []
    cluster_list = []
    train = training_df[['id', 'belong']]
    for i in range(1, 11):
        if len(str(i)) == 1:
            training = train['id'].loc[train['belong'] == "Wei Wang 000" + str(i)].as_matrix()
        else:
            training = train['id'].loc[train['belong'] == "Wei Wang 00" + str(i)].as_matrix()
        training_list = list(combinations(training, 2))
        total_list += training_list
    len(total_list)
    pre = pre_df[['id', 'result']]
    for i in range(0, 10):
        predict = pre['id'].loc[pre['result'] == i].as_matrix()
        s = list(combinations(predict, 2))
        cluster_list += s
    len(cluster_list)
    count = 0
    for i in cluster_list:
        if i in total_list:
            count = count + 1
    print("准确率为：")
    print(count/len(total_list))

#计算rand指数函数
def rand_index(training_df,pre_df):
    total_list = []
    cluster_list = []
    train = training_df[['id', 'belong']]
    all=training_df[['id']].as_matrix()
    all_list=list(combinations(all,2))
    for i in range(1, 11):
        if len(str(i)) == 1:
            training = train['id'].loc[train['belong'] == "Wei Wang 000" + str(i)].as_matrix()
        else:
            training = train['id'].loc[train['belong'] == "Wei Wang 00" + str(i)].as_matrix()
        training_list = list(combinations(training, 2))
        total_list += training_list
    len(total_list)
    pre = pre_df[['id', 'result']]
    for i in range(0, 10):
        predict = pre['id'].loc[pre['result'] == i].as_matrix()
        s = list(combinations(predict, 2))
        cluster_list += s
    len(cluster_list)
    count = 0
    for i in cluster_list:
        if i in total_list:
            count = count + 1
    print(count)
    d=0
    for j in all_list:
        if j not in total_list and j not in cluster_list:
            d=d+1

    print("准确率为：")
    print((count+d )/ 94395)


def main():
    preprocess()
    #to_txt()
    #dataset=load_dataset("dblp.txt")
    print("************************************************")
    print("1.TF_IDF+K-Means"+'\n')
    print("2.doc2vec+K-Means"+'\n')
    print("3.TF-IDF+GMM"+'\n')
    print("4.TF-IDF+birch" + '\n')
    print("************************************************")
    choice=input("Enter your choice")
    if choice=="1":
        to_txt(2)
        dataset = load_dataset("dblp.txt")
        result=kmeans_tfidf(dataset,False)
    elif choice=="2":
        to_doc()
        result=kmeans_doc2vec("doc.txt")
    elif choice=="3":
        to_txt(3)
        dataset = load_dataset("dblp.txt")
        result=gmm_tfidf(dataset)
    elif choice=="4":
        to_txt(4)
        dataset = load_dataset("dblp.txt")
        result=birch_tfidf(dataset)
    else:
        print("input error!")
    pre_df=get_result(result)
    #rand_index(get_df("dblp.csv"),pre_df)# rand 指数
    precision_rate(get_df("dblp.csv"),pre_df)#老师讲的验证方法

if __name__ == '__main__':
    main()

