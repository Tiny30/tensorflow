'''
Desc1：这是一个用tensorflow张量的方式实现深度网络前向传播的代码
Desc2：激活函数调用activation激活函数类中的激活函数方法
Author:SQY
DateTime:2020-7-15
'''
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
import os
from activation import Activate
#类的实例化
act = Activate()
# 去除打印无关信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 下载数据
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
# 将数据格式进行转变为张量,并且将x的范围归一到0-1范围
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
#测试数据集划分
x_test = tf.convert_to_tensor(x_test,dtype=tf.float32)/255
y_test = tf.convert_to_tensor(y_test,dtype=tf.int32)
# print("x_train:", x_train.shape, x_train.dtype)
# print("y_train:", y_train.shape, y_train.dtype)
# minist数据集是【6000,28，28】，现在对minist第一个维度切分，并且每次取128
x_train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(128)
# # 不断迭代
# x_train_iter = iter(x_train_db)
# # 返回迭代的数据
# sample = next(x_train_iter)
# print("size:", sample[0].shape, sample[1].shape)

# 定义w，b，其中w[input_dims,output_dims],b[output_dims]
# w[784,256]-->[256,128]-->[128,10]
# b初始化为0
w1 = tf.Variable( tf.random.truncated_normal([784, 256],stddev = 0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128],stddev = 0.1 ))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10],stddev = 0.1))
b3 = tf.Variable(tf.zeros([10]))

# 求h = w @ x +b:w@x矩阵乘法
lr = 1e-3
for epoch in range(10):#针对每个数据集迭代10次
    for step,(x_train, y_train) in enumerate(x_train_db):#针对每轮128个数据迭代求权重，偏移，求导
        # 改变x的维度（-1,28*28）
        x_train = tf.reshape(x_train, [-1, 28 * 28])
        #梯度求导，主要对w1b1w2b2w3b3
        with tf.GradientTape()  as tape:
            h1 = tf.matmul(x_train, w1) + b1
            act.Relu(h1)
            h2 = tf.matmul(h1, w2) + b2
            act.Relu(h2)
            output = tf.matmul(h2, w3) + b3

            # 计算均方差损失函数
            # 将y转化为向量one_hot向量
            y_train = tf.one_hot(y_train, 10)
            loss = tf.square(y_train - output)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
        # 根据上面的梯度求导来更新w，b的值
        #w1 = w1 - lr * w1_grads
            # w1 = w1 - lr * grads[0]
            # b1 = b1 - lr * grads[1]
            # w2 = w2 - lr * grads[2]
            # b2 = b2 - lr * grads[3]
            # w3 = w3 - lr * grads[4]
            # b3 = b3 - lr * grads[5]
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1] )
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step %100 == 0:
            print("loss:",float(loss),"step:",step)
    print("epoch:",epoch)

    #初始化正确的数量，初始化总的测试样本的数量
    total_correct = 0
    total_num = 0
    #用测试数据计算准确率
    for step,(x_test,y_test) in enumerate(test_db):
        x_test = tf.reshape(x_test,[-1,28*28])
        h1 = tf.nn.relu(tf.matmul(x_test,w1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1,w2) + b2)
        output = tf.matmul(h2,w3) + b3

        #用softmax函数求出输出的概率
        prob = tf.nn.softmax(output,axis=1)
        #求出输出概率最大值的索引
        pred = tf.argmax(prob,axis=1)
        #pred为int64，需要人为转化为int32
        pred = tf.cast(pred,dtype=tf.int32)
        #计算准确率
        #将预测值与真实值进行比较，并转化为int32类型转（0-1）
        correct = tf.cast(tf.equal(pred,y_test),dtype=tf.int32)
        #统计预测对的个数
        correct = tf.reduce_sum(correct)
        #correct是一个tensor，需要转化为int类型
        total_correct += int(correct)
        total_num += x_test.shape[0]
    #计算accuracy，total_correct / total_num
    acc = total_correct / total_num
    print("acc:",float(acc))




# 输出x，y最大值最小值
# print("x_max:",tf.reduce_max(x_train))
# print("x_min:",tf.reduce_min(x_train))
# print("y_max:",tf.reduce_max(y_test))
