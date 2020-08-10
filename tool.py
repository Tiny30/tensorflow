import tensorflow as tf
import numpy as np

class Tool_activation:
    def __int__(self,x):
        self.x = x

    #激活函数
    #sigmoid函数
    def sigmoid(self,x):
        return  1 / 1 + np.exp(-x)
    #tanh
    def tanh(self,x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    #Relu
    def relu(self,x):
        return np.maximum(0,x)
    #LeakyRelu
    def leakyRelu(self,x):
        return np.maximum(0.01 * x,x)
    def softMax(self,x):
        return tf.exp(x) / tf.reduce_sum(tf.exp(x))
