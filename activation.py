'''
Desc1：这是一个activation激活函数类，封装了sigmoid，tanh，Relu，LeakyRelu等激活函数
Author:SQY
DateTime:2020-7-16
'''
import numpy as np
import tensorflow as tf

class Activate:

    #定义一个sigmoid激活函数
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    #定义一个tanh激活函数
    def tanh(self,x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    #定义一个relu激活函数
    def Relu(self,x):
        return np.maximum(0,x)

    #定义一个LeakyRelu激活函数:对于小于0的值去0.01*x，大于0的值去x
    def LeakyRelu(self,x):
        return  np.maximum(0.01*x,x)


