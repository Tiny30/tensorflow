'''
Desc：1.这是一个自定义神经网络，该网络没有偏置参数
      2.自定义网络层
      3.自定义网络
      4.数据为CIFAR10
Target:实现目标检测识别
Author:SQY
Date：2020-07-22
'''

import tensorflow as tf
from tensorflow.keras import layers,optimizers,losses,datasets,metrics,Sequential,models
import os
from tensorflow import keras

os.environ["TF_CPP_WIN_LOG_LEVEL"] = '2'
#定义一个数据处理的方法
def process(x_train,y_train):
    x_train = tf.cast(x_train,dtype = tf.float32) / 255
    y_train = tf.cast(y_train,dtype = tf.int32)
    return x_train,y_train


#加载数据
batchsize = 128
(x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()
y_train = tf.squeeze(y_train)
y_train = tf.one_hot(y_train,10)

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.map(process).shuffle(10000).batch(batchsize)
#定义神经网络
class MyDense(layers.Dense):
    def __init__(self,input_dims,output_dims):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w',[input_dims,output_dims])
        self.bias = self.add_variable('b',[output_dims])
    def call(self, inputs,training = None):
        x  = tf.matmul(inputs,self.kernel)
        return x

#定义网络运行顺序
class MyModel(keras.models):
    def __init__(self):
        super(MyModel, self).__init__()
        # [3072,784][784,256][256,128][128,64][64,10]
        self.fc1 = MyDense(32 * 32 * 3,784)
        self.fc2 = MyDense(784,256)
        self.fc3 = MyDense(256,128)
        self.fc4 = MyDense(128,64)
        self.fc5 = MyDense(64,10)
    def __call__(self,inputs,training = None):
        #reshape
        x = tf.reshape(inputs,[-1,32*32*3])
        x = self.fc1(x)
        #添加激活函数
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x


model = MyModel()
model.compile(optimizers = optimizers.Adam(1e-3),
              losses = tf.losses.categorical_crossentropy(from_logits=True),
              metrics = ['accuracy']
              )
model.fit(train_db,epochs = 1)
model.evaluate("This is evaluate:",x_test,y_test)
model.predict("This is predict:",x_test)
model.save('customize.h5')
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
#
# #定义一个数据处理方法
# def process(x_train,y_train):
#     x_train = tf.cast(x_train,dtype=tf.float32) /255
#     y_train = tf.cast(y_train,dtype= tf.int32)
#     return x_train,y_train

#加载CIFAR10数据集 [500000,32,32,3]
#产生y不在是【10】，而是[10,1],所以需要将1给去除
(x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()
y_train = tf.squeeze(y_train)
y = tf.one_hot(y_train,10)
#
# #数据切分
# train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
# #转化为map,用shuffle打乱，设置每轮载入数据量
# train_db = train_db.map(process).shuffle(10000).batch(128)
#
#
# #自定义Dense类实现自定义网络层
# class MyDense(layers.Layer):
#     def __init__(self,input_dims,output_dims):
#         super(MyDense, self).__init__()
#         self.kernel = self.add_variable('w',[input_dims,output_dims])
#         # self.bias = self.add_variable('b',[output_dims])
#     def call(self, inputs, training = None):
#         #实现网络逻辑，x = input @ kernel (s = wx )
#         x = tf.matmul(inputs,self.kernel)
#         return x
#
# #自定义网络
# class MyModel(keras.models):
#     def __init__(self):
#         super(MyModel, self).__init__()
#     def __call__(self, input, training = None):
#         pass