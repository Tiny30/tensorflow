import tensorflow as tf
import numpy as np
a = tf.convert_to_tensor(np.ones([2,3]))
b = tf.random_normal([2,3],1,1)
tf.gather()
tf.expand_dims()
tf.squeeze()
tf.broadcast_dynamic_shape()
tf.broadcast_to(a)
tf.split(a,axis=0,num_or_size_splits=2)


