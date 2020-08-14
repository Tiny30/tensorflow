'''
Desc:这是一个多层的LSTM神经网络对文本数据进行二分类情感处理，主要分为积极和消极
Author:SQY
Date:2020-8-10
'''
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
os.environ['TP_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')
class Params():
    NUM_WORD = 10000
    UNITS = 64
    BATCH_SIZE = 128
    MAX_LEN = 80
    EMBEDDING_DIM = 100
    EPOCH = 10
    LEARN_RATE = 0.003

class Data2vec(Params):
    def read_data(self, path):
        train = pd.read_csv(path + "train.tsv", sep='\t', encoding='utf8')
        test = pd.read_csv(path + 'test.tsv', sep='\t', encoding='utf8')
        x_train = train['text_a']
        y_train = train['label']
        x_test = test['text_a']
        y_test = test['label']
        return x_train, x_test, y_train, y_test

    def data_process(self, x_train, x_test, y_train, y_test):
        tokenizer = Tokenizer(Params.NUM_WORD)
        tokenizer.fit_on_texts(x_train)
        tokenizer.fit_on_texts(x_test)
        x_train = tokenizer.texts_to_sequences(x_train)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_train = pad_sequences(x_train, maxlen=Params.MAX_LEN)
        x_test = pad_sequences(x_test, maxlen=Params.MAX_LEN)
        db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch( batch_size=Params.BATCH_SIZE,drop_remainder=True)
        db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000).batch(batch_size=Params.BATCH_SIZE,drop_remainder=True)
        return db_train, db_test, x_train, x_test

'''
Desc:这是一个自定义的LSTM网络层
'''
class My_lstm(keras.Model, Params):
    def __init__(self, units):
        super(My_lstm, self).__init__()
        self.state0 = [tf.zeros([Params.BATCH_SIZE, units]), tf.zeros([Params.BATCH_SIZE, units])]
        self.state1 = [tf.zeros([Params.BATCH_SIZE, units]), tf.zeros([Params.BATCH_SIZE, units])]
        self.embedding = layers.Embedding(Params.NUM_WORD, Params.EMBEDDING_DIM, input_length=Params.MAX_LEN)
        self.lstmcell0 = layers.LSTMCell(units=units, dropout=Params.DROP_OUT)
        self.lstmcell1 = layers.LSTMCell(units=units, dropout=Params.DROP_OUT)
        self.outlayers = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.embedding(x)
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):
            out0, state0 = self.lstmcell0(word, state0, training)
            out, state1 = self.lstmcell1(out0, state1, training)
        x = self.outlayers(out)
        prob = tf.sigmoid(x)
        return prob


'''
Desc:这是使用tf.keras中的Sequential模块
'''
class LstmSequential(keras.Model,Params):
    def __init__(self,units):
        super(LstmSequential, self).__init__()
        self.embedding = layers.Embedding(Params.NUM_WORD,Params.EMBEDDING_DIM,input_length=Params.MAX_LEN)
        self.lstm = keras.Sequential([layers.LSTM(units=units,return_sequences=True,unroll=True),
                                      keras.layers.Dropout(rate=0.5),
                                      layers.LSTM(units=units,unroll=True),
                                      keras.layers.Dropout(rate=0.5),
                                      ])
        self.outlayer = layers.Dense(1)
    def call(self, inputs, training=None):
        x = inputs
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.outlayer(x)
        prob = tf.sigmoid(x)
        return prob
if __name__ == '__main__':
    path = 'D:\\PyCharm\\tensorflow\\data\\'
    data = Data2vec()
    x_train, x_test, y_train, y_test = data.read_data(path)
    db_train, db_test, x_train, x_test = data.data_process(x_train, x_test, y_train, y_test)
    model = LstmSequential(Params.UNITS)
    model.compile(loss=tf.losses.BinaryCrossentropy(),
                  optimizer=tf.optimizers.Adamax(Params.LEARN_RATE),
                  metrics=['accuracy'])
    model.fit(db_train,epochs=Params.EPOCH,validation_data=db_test)
    model.evaluate(db_test)

