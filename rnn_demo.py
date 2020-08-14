'''
Desc:这是一个单层的rnn神经网络对文本数据进行二分类情感处理，主要分为积极和消极
Author:SQY
Date:2020-8-10
'''
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os
os.environ['TP_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

#定义参数
num_words = 35000 #在文本中，常见的词在一万左右，剩下不常见的我们基本用不到，此地将不常见的词均用10000表示
max_len = 80 #设置每个句子的长度
batch_size = 128#每批次训练多少
embedding_dim = 100#词向量的维度



def split_data():
    # 加载数据
    path = 'D:\\PyCharm\\tensorflow\\data\\'
    train = pd.read_csv(path + 'train.tsv', sep='\t', encoding='utf8').replace('\n', '', regex=True)
    test = pd.read_csv(path + 'test.tsv', sep='\t', encoding='utf8').replace('\n', '', regex=True)
    x_train = train['text_a']
    y_train = train['label']
    x_test = test['text_a']
    y_test = test['label']
    tokenizer = Tokenizer(num_words=100000,lower=False)
    tokenizer.fit_on_texts(x_train)
    tokenizer.fit_on_texts(x_test)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_train = pad_sequences(x_train,maxlen=max_len)
    x_test = pad_sequences(x_test,maxlen=max_len)
    db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    db_train = db_train.shuffle(1000).batch(batch_size = batch_size,drop_remainder=True)
    db_test = db_test.shuffle(1000).batch(batch_size = batch_size,drop_remainder=True)
    print(x_train.shape,tf.reduce_min(y_train),tf.reduce_max(y_train))
    print(x_test.shape)
    return db_train,db_test,x_train,x_test
class MyRnn(keras.Model):
    def __init__(self,units):
        super(MyRnn, self).__init__()
        #设定初始状态
        self.state0 = [tf.zeros([batch_size,units])]
        self.embedding = layers.Embedding(num_words,embedding_dim,input_length=max_len)
        self.rnncell0 = layers.SimpleRNNCell(units,dropout=0.3)
        self.outlayer = layers.Dense(1)
    def call(self, inputs, training=None):
        x = inputs
        x = self.embedding(x)
        state0 = self.state0
        for word in tf.unstack(x,axis=1):
            output,state1 = self.rnncell0(word,state0,training)
            state0 = state1
        x = self.outlayer(output)
        prob = tf.sigmoid(x)
        return prob
def main():
    units = 64
    model = MyRnn(units)
    model.compile(optimizer=tf.optimizers.Adam(0.002),
                  loss = tf.losses.binary_crossentropy,
                  metrics=["accuracy"])
    db_train,db_test,x_train,x_test= split_data()
    model.fit(db_train,epochs=10)
    print('eva:',model.evaluate(db_test))
    # pre = model.predict(x_test)
    # print('pre:',pre)
if __name__ == '__main__':
    main()