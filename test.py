import tensorflow as tf
from tensorflow.keras import datasets
from tool import Tool_activation
t_act = Tool_activation()

#加载数据
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()

#将数据集转化为张量
x_train = tf.convert_to_tensor(x_train,dtype=tf.float32)/255
y_train = tf.convert_to_tensor(y_train,dtype=tf.int32)
x_test = tf.convert_to_tensor(x_test,dtype=tf.float32)/255
y_test = tf.convert_to_tensor(y_test,dtype=tf.int32)
#把数据集切分为128个为一个训练组
x_train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(128)
#迭代测试一下
# x_train_iter = iter(x_train_db)
# sample = next(x_train_iter)
# print("size:",sample[0].shape)

#定义w，b
#w[input_dims,output_dims] b[output_dims]
#w[784,256] -> [256,128] -->[128,10]   b[256] -->[128]  -->[10]
w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

#计算每一轮的 h = w@x+b,限定没多少轮打印一次
lr = 1e-3
for epoch in range(11):
    print("epoch:",epoch)
    for step,(x_train,y_train) in enumerate(x_train_db):
        #先将x_train改变维度
        x_train = tf.reshape(x_train,[-1,28 * 28])
        with tf.GradientTape() as tape:
            h1 = tf.matmul(x_train,w1) + b1
            #激活函数
            t_act.leakyRelu(h1)
            h2 = tf.matmul(h1,w2) + b2
            t_act.leakyRelu(h1)
            output = tf.matmul(h2,w3) + b3
            #将y向量化
            y_true = tf.one_hot(y_train,10)
            loss = tf.reduce_mean(tf.square(y_true - output))
            grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
        #求导根据loss不断更新w,b
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
        if step %100 ==0:
            print("step:",step,",loss:",float(loss))

    for step,(x_test,y_test) in enumerate(test_db):
        x_test = tf.reshape(x_test,[-1,28 * 28])
        h1 = tf.nn.leaky_relu(tf.Variable(tf.matmul(x_test,w1) + b1))
        h2 = tf.nn.leaky_relu(tf.Variable(tf.matmul(h1,w2) + b2))
        output = tf.Variable(tf.matmul(h2,w3) + b3)
        prob = tf.nn.softmax(output,axis=1)



















































