'''
Desc:这是一个word2vec+lstm写的一个文本的情感二分类算法
    1.word2vec直接调用gensimku
    2.lstm使用的是tensorflow2.0中的keras模块，使用了四层网络层
Author:SQY
Date；2020-8-12
'''
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import jieba

'''
配置参数
'''


class Config():
    pass


'''
对数据进行预处理
    1.读取数据
    2.分词
    3.去除停用词
'''


class DataProcess():
    '''
    读取文本数据
    '''

    def read_data(self, path):
        train = pd.read_csv(path + "train.tsv", sep='\t', encoding='utf8')
        test = pd.read_csv(path + 'test.tsv', sep='\t', encoding='utf8')
        x_train = train['text_a']
        y_train = train['label']
        x_test = test['text_a']
        y_test = test['label']

        return x_train, y_train, x_test, y_test

    '''
    将文本数据分词（jieba分词）
    '''

    def split_word(self, sentence):
       return jieba.lcut(sentence)

    '''
    将分词后的文本去除停用词
    
    '''

    def stop_word(self, path,sentence):
        stop_word = []
        sentence_out = []
        #读取停用词
        with open(path + "stop_words.txt",'r',encoding='utf8') as fp:
            for line in fp.readlines():
                stop_word.append(line)
        #去除停用词，遍历文本，如果文本中的词不在停用词中且不为空格则用一个新的数组存起来
        for word in sentence:
            if word not in stop_word:
                if word != " ":
                    sentence_out = sentence_out.append(word)
        return sentence_out


'''
构建词向量模型
'''


class W2Vec():
    pass


'''
构建神经网络模型
'''


class MyRnn(keras.Model):
    def __init__(self):
        super(MyRnn, self).__init__()
        self.embedding = layers.Embedding()

    def call(self, inputs, training=None, mask=None):
        pass


'''
入口函数
'''
if __name__ == '__main__':
    path = 'D:\\PyCharm\\tensorflow\\data\\'
    data = DataProcess()
    x_train, y_train, x_test, y_test = data.read_data(path)
    sentence  = x_train['text_a']
    sentence = data.split_word(x_train)
    x_train = data.stop_word(path,sentence = sentence)
    print(x_train)
