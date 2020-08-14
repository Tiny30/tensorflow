

# *- coding: utf-8 -*-

# =================================
# time: 2020.8.9
# author: @tangzhilin
# function: Bert模型训练流程
# =================================

from abc import ABC
import pandas as pd
import tensorflow as tf
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules import Module
from transformers import *
from datetime import datetime
test = pd.read_csv("D:\\PyCharm\\tensorflow\\data\\t.tsv",sep="\t",encoding='utf8').dropna(0)
strategy = tf.distribute.MirroredStrategy()


class CustomDataset(Dataset):
    """
    function: 数据准备
    """
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        # self.data = dataframe
        # print(self.data['text_a'])
        self.comment_text = dataframe['text_a']
        self.targets = self.data['label']
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split(sep ='\n'))

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class BertClass(tf.keras.Model, ABC):
    """
    function: 模型
    """
    # Model 需要def call()调用
    def __init__(self):
        super(BertClass, self).__init__()
        # self.bert_1 = AlbertForMaskedLM.from_pretrained("voidful/albert_chinese_tiny")
        self.bert_1 = TFBertModel.from_pretrained("bert-base-chinese")
        self.l2 = tf.keras.layers.Dropout(rate=0.3)
        self.l3 = tf.keras.layers.Dense(768, activation='relu')
        self.l4 = tf.keras.layers.Dense(2, activation='sigmoid')

    @tf.function
    def call(self, ids, mask, token_type_ids):
        _, output_1 = self.bert_1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        # print("output_2: ", output_2.shape, output_2)
        output_3 = self.l3(output_2)
        # print("output_3: ", output_3.shape, output_3)
        output_4 = self.l4(output_3)

        return output_4


with strategy.scope():
    def loss_fn(target, output):
        """
        function: 获取损失值
        :param target: label
        :param output: prediction
        :return: loss
        """
        target = tf.convert_to_tensor(target)
        output = tf.convert_to_tensor(output)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return loss(y_true=target, y_pred=output)


with strategy.scope():
    def accuracy_fn(target, output):
        """
        function: 获取准确值
        :param target: label
        :param output: prediction
        :return: accuracy
        """
        target = tf.convert_to_tensor(target)
        output = tf.convert_to_tensor(output)

        return tf.keras.metrics.\
            sparse_categorical_crossentropy(y_true=target, y_pred=output, from_logits=True)


with strategy.scope():
    def train():
        """
        function: 开启模型训练过程
        :return: Model, Log
        """
        time_month = datetime.now().month
        time_day = datetime.now().day
        device = 'cuda' if cuda.is_available() else 'cpu'
        # token = BertTokenizer.from_pretrained("voidful/albert_chinese_tiny")
        token = BertTokenizer.from_pretrained("bert-base-chinese")
        model_save_path = '../Transformers/Bert_Model/{}-{}-model'.format(time_month, time_day)

        # 创建日志文件的目录
        log_dir = '../Transformers/Bert_Model/{}-{}-log'.format(time_month, time_day)
        summary_writer = tf.summary.create_file_writer(log_dir)

        model = BertClass()
        tf.device(device)
        optimizer = tf.keras.optimizers.Adam(lr=1e-05)
        train_set = CustomDataset(test, token, 400)
        train_params = {
            'batch_size': 2,
            'shuffle': True,
            'num_workers': 0
        }

        train_set_pt = DataLoader(train_set, **train_params)

        def train_step(model_, id_, mk_, type_ids_, optimizer_, target_):
            with tf.GradientTape() as tp:
                y_pred = model_(id_, mk_, type_ids_)
                loss_value = loss_fn(target=target_, output=y_pred)
                acc = accuracy_fn(target_, y_pred)

            gradient = tp.gradient(loss_value, model.trainable_variables)

            optimizer_.apply_gradients(zip(gradient, model.trainable_variables))

            return loss_value, acc

        for _, batch_data in enumerate(train_set_pt, 0):
            ids = tf.convert_to_tensor(batch_data['ids'].detach().numpy())
            print("ids: ", ids[0])
            print("decode: ", token.decode(ids[0]))
            print("lenth: ", len(ids[0]))
            mask = tf.convert_to_tensor(batch_data['mask'].detach().numpy())
            token_type_ids = tf.convert_to_tensor(batch_data['token_type_ids'].detach().numpy())
            targets = tf.convert_to_tensor(batch_data['targets'].detach().numpy())
            # 相当于tf的 tf.compat.v1.get_default_xxx重置模型的优化效果
            loss, accuracy = train_step(model_=model, id_=ids, mk_=mask,
                                        type_ids_=token_type_ids, optimizer_=optimizer,
                                        target_=targets)
            # 将loss和accuracy写入日志文件
            # with summary_writer.as_default():
            #     tf.summary.scalar(name="loss_value_step:{}".format(_),
            #                       data=loss, step=_)
            # with summary_writer.as_default():
            #     tf.summary.scalar(name='accuracy_value_step:{}'.format(_),
            #                       data=accuracy.numpy().mean(), step=_)
            print(loss, accuracy)

        tf.saved_model.save(model, export_dir=model_save_path)

train()

















#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
# # *- coding: utf-8 -*-
#
# # =================================
# # time: 2020.8.9
# # author: @tangzhilin
# # function: Bert模型训练流程
# # =================================
#
# from abc import ABC
# import pandas as pd
# import tensorflow as tf
#
# # import torch
# # from torch import cuda
# # from torch.utils.data import Dataset, DataLoader
# # from torch.nn.modules import Module
# from transformers import *
# from datetime import datetime
# test = pd.read_csv("D:\\PyCharm\\tensorflow\\data\\t.tsv",'r', engine='python',encoding='utf8',error_bad_lines=False)
# strategy = tf.distribute.MirroredStrategy()
#
#
# class CustomDataset():
#     """
#     function: 数据准备
#     """
#     def __init__(self, dataframe, tokenizer, max_len):
#         self.tokenizer = tokenizer
#         self.data = dataframe
#         self.comment_text = self.data["text_a"]
#         self.targets = self.data['label']
#         self.max_len = max_len
#
#     def __len__(self):
#         return len(self.comment_text)
#
#     def __getitem__(self, index):
#         comment_text = str(self.comment_text[index])
#         comment_text = " ".join(comment_text.split())
#
#         inputs = self.tokenizer.encode_plus(
#             comment_text,
#             None,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             pad_to_max_length=True,
#             return_token_type_ids=True
#         )
#         ids = inputs['input_ids']
#         mask = inputs['attention_mask']
#         token_type_ids = inputs["token_type_ids"]
#
#         return {
#             'ids':tf.convert_to_tensor(ids,dtype=tf.int32),
#             'mask': tf.convert_to_tensor(mask,dtype=tf.int32),
#             'token_type_ids': tf.convert_to_tensor(token_type_ids,dtype=tf.int32),
#             'targets': tf.convert_to_tensor(self.targets[index], dtype=tf.in32),
#
#             # 'ids': torch.tensor(ids, dtype=tf.in32),
#             # 'mask': torch.tensor(mask, dtype=tf.in32),
#             # 'token_type_ids': torch.tensor(token_type_ids, dtype=tf.in32),
#             # 'targets': torch .tensor(self.targets[index], dtype=tf.in32)
#         }
#
#
# class BertClass(tf.keras.Model, ABC):
#     """
#     function: 模型
#     """
#     # Model 需要def call()调用
#     def __init__(self):
#         super(BertClass, self).__init__()
#         # self.bert_1 = AlbertForMaskedLM.from_pretrained("voidful/albert_chinese_tiny")
#         self.bert_1 = TFBertModel.from_pretrained("bert-base-chinese")
#         #self.bert_1 = TFBertModel.from_pretrained("bert-base-uncased")
#         self.l2 = tf.keras.layers.Dropout(rate=0.3)
#         self.l3 = tf.keras.layers.Dense(768, activation='relu')
#         self.l4 = tf.keras.layers.Dense(2, activation='sigmoid')
#
#     @tf.function
#     def call(self, ids, mask, token_type_ids):
#         _, output_1 = self.bert_1(ids, attention_mask=mask, token_type_ids=token_type_ids)
#         output_2 = self.l2(output_1)
#         # print("output_2: ", output_2.shape, output_2)
#         output_3 = self.l3(output_2)
#         # print("output_3: ", output_3.shape, output_3)
#         output_4 = self.l4(output_3)
#
#         return output_4
#
#
# with strategy.scope():
#     def loss_fn(target, output):
#         """
#         function: 获取损失值
#         :param target: label
#         :param output: prediction
#         :return: loss
#         """
#         target = tf.convert_to_tensor(target)
#         output = tf.convert_to_tensor(output)
#         loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#         return loss(y_true=target, y_pred=output)
#
#
# with strategy.scope():
#     def accuracy_fn(target, output):
#         """
#         function: 获取准确值
#         :param target: label
#         :param output: prediction
#         :return: accuracy
#         """
#         target = tf.convert_to_tensor(target)
#         output = tf.convert_to_tensor(output)
#
#         return tf.keras.metrics.\
#             sparse_categorical_crossentropy(y_true=target, y_pred=output, from_logits=True)
#
#
# with strategy.scope():
#     def train():
#         """
#         function: 开启模型训练过程
#         :return: Model, Log
#         """
#         time_month = datetime.now().month
#         time_day = datetime.now().day
#         token = BertTokenizer.from_pretrained("bert-base-chinese")
#         # token = BertTokenizer.from_pretrained("bert-base-uncased")zhe
#         model_save_path = '../Transformers/Bert_Model/{}-{}-model'.format(time_month, time_day)
#
#         # 创建日志文件的目录
#         log_dir = '../Transformers/Bert_Model/{}-{}-log'.format(time_month, time_day)
#         summary_writer = tf.summary.create_file_writer(log_dir)
#
#         model = BertClass()
#         optimizer = tf.keras.optimizers.Adam(lr=1e-05)
#         train_set = CustomDataset(test, token, 400)
#         tf.summary(train_set)
#         # train_params = {
#         #     'batch_size': 2,
#         #     'shuffle': True,
#         #     'num_workers': 0
#         # }
#         train_set_pt = tf.data.Dataset.from_tensor_slices(train_set).shuffle(1000).batch(2,drop_remainder=True)
#         # train_set_pt = DataLoader(train_set, **train_params)
#
#         def train_step(model_, id_, mk_, type_ids_, optimizer_, target_):
#             with tf.GradientTape() as tp:
#                 y_pred = model_(id_, mk_, type_ids_)
#                 loss_value = loss_fn(target=target_, output=y_pred)
#                 acc = accuracy_fn(target_, y_pred)
#
#             gradient = tp.gradient(loss_value, model.trainable_variables)
#
#             optimizer_.apply_gradients(zip(gradient, model.trainable_variables))
#
#             return loss_value, acc
#
#         for _, batch_data in enumerate(train_set_pt, 0):
#             ids = tf.convert_to_tensor(batch_data['ids'].detach().numpy())
#             print("ids: ", ids[0])
#             print("decode: ", token.decode(ids[0]))
#             print("lenth: ", len(ids[0]))
#             mask = tf.convert_to_tensor(batch_data['mask'])
#             token_type_ids = tf.convert_to_tensor(batch_data['token_type_ids'])
#             targets = tf.convert_to_tensor(batch_data['targets'])
#             # 相当于tf的 tf.compat.v1.get_default_xxx重置模型的优化效果
#             loss, accuracy = train_step(model_=model, id_=ids, mk_=mask,
#                                         type_ids_=token_type_ids, optimizer_=optimizer,
#                                         target_=targets)
#             # 将loss和accuracy写入日志文件
#             # with summary_writer.as_default():
#             #     tf.summary.scalar(name="loss_value_step:{}".format(_),
#             #                       data=loss, step=_)
#             # with summary_writer.as_default():
#             #     tf.summary.scalar(name='accuracy_value_step:{}'.format(_),
#             #                       data=accuracy.numpy().mean(), step=_)
#             print(loss, accuracy)
#
#         tf.saved_model.save(model, export_dir=model_save_path)
#
# train()
#
