'''
Desc"计算top_k_accuracy,top_k_accuracy通常用来求分类为题的准确率
Author:SQY
Date:2020-7-16
'''
import  tensorflow as tf
class top_k_acc:
    '''
    output:输出的样本
    target：目标样本
    topk：计算top1，还是topk2...  topk = (1,2,3)计算top1到top3的accuracy
    '''
    def compute_topk_acc(self,output,target,topk = (1)):
        #去topk最大值
        max_k = max(topk)
        #获取target的长度
        data_len = target.shpae[0]
        #获取预测数据（output）的top_k的索引值
        pred = tf.math.top_k(output,max_k).indices
        #转置
        pred = tf.transpose(pred,perm = [1,0])
        #target扩张
        target_ = tf.broadcast_to(target,pred.shape)
        #真实值比较
        correct = tf.equal(pred,target_)

        #循环统计
        res = []
        for k in topk:
            correct_k = tf.cast(tf.reshape(correct[:k],[-1]),dtype=tf.float32)
            correct_k = tf.reduce_sum(correct_k)
            acc = float(correct_k/data_len)
            res.append(acc)
        return res