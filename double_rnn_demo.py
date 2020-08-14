'''
Desc:这是一个多层的rnn神经网络对文本数据进行二分类情感处理，主要分为积极和消极
Author:SQY
Date:2020-8-10
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
class Params():
    NUM_WORDS = 10000
    EMBEDDING_DIM = 100
    BATCH_SIZE = 128
    MAX_LEN = 80
    EPOCH = 10
    UNITS = 64
class DataSplit():
    def read_data(self,path):
        train = pd.read_csv(path + "train.tsv",sep="\t",encoding='utf8')
        test = pd.read_csv(path + "test.tsv",sep='\t',encoding='utf8')
        x_train = train['text_a']
        y_train = train['label']
        x_test = test['text_a']
        y_test = test['label']
        return
    def data2vec(self,x_train,y_train,x_test,y_test,Params):
        tokenizer = Tokenizer(Params.NUM_WORDS)
        tokenizer.fit_on_texts(x_train)
        tokenizer.fit_on_texts(x_test)
        x_train = tokenizer.texts_to_sequences(x_train)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_train = pad_sequences(x_train,maxlen=Params.MAX_LEN)
        x_test = pad_sequences(x_test,maxlen=Params.MAX_LEN)
        db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(1000).batch(Params.BATCH_SIZE,drop_remainder=True)
        db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test)).shuffle(1000).batch(Params.BATCH_SIZE,drop_remainder=True)
        return db_train,db_test,x_train,x_test
'''
Desc:这是一个自定义rnn网络
'''
class MyRnn(keras.Model):
    def __init__(self,Params,units):
        super(MyRnn, self).__init__()
        self.state0 = [tf.zeros([Params.BATCH_SIZE,units])]
        self.state1 = [tf.zeros([Params.BATCH_SIZE,units])]
        self.embedding = layers.Embedding(Params.NUM_WORDS,Params.EMBEDDING_DIM,input_length=Params.MAX_LEN)
        self.rnncell0 = layers.SimpleRNNCell(units,dropout=0.3)
        self.rnncell1 = layers.SimpleRNNCell(units,dropout=0.3)
        self.outlayer = layers.Dense(1)


    def call(self, inputs, training=None):
        x = inputs
        x = self.embedding(x)
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x,axis=1):
            out_layer,state0 = self.rnncell0(word,state0,training)
            out,state1 = self.rnncell1(out_layer,state1,training)
        x = self.outlayer(out)
        prob = tf.sigmoid(x)
        return prob

'''
Desc:这是一个使用tensorflow中自带rnn层定义的网络
'''
class Rnn_layer(keras.Model):
    def __init__(self, Params, units):
        super(Rnn_layer, self).__init__()
        self.embedding = layers.Embedding(Params.NUM_WORDS, Params.EMBEDDING_DIM, input_length=Params.MAX_LEN)
        self.rnn = keras.Sequential([layers.SimpleRNN(units= units,dropout=0.3,return_sequences=True),
                                     layers.SimpleRNN(units= units,dropout=0.3)])
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.outlayer(x)
        prob = tf.sigmoid(x)
        return prob
if __name__ == '__main__':
    path = 'D:/PyCharm/tensorflow/data/'
    data = DataSplit()
    x_train,y_train,x_test,y_test = data.read_data(path)
    db_train,db_test,x_train,x_test = data.data2vec(x_train,y_train,x_test,y_test,Params)
    # model = MyRnn(Params,units=Params.UNITS)
    model = Rnn_layer(Params,units= Params.UNITS)
    model.compile(optimizer=tf.optimizers.Adam(0.02),
                  loss=tf.losses.binary_crossentropy,
                  metrics=['accuracy'])
    model.fit(db_train,epochs=Params.EPOCH)
    eva = model.evaluate(db_test)
    # print(model.predict(x_test))