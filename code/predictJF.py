# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from code.selfattention_CNN import TextCNN, TCNNConfig


try:
    bool(type(unicode))
except NameError:
    unicode = str


class CnnModel:
    def __init__(self,save_path,sess):
        self.config = TCNNConfig()
        self.model = TextCNN(self.config)

        self.session = sess
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, input1, input2):#input1是事实或者结论，input2是法条，
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运

        feed_dict = {
            self.model.input_x_1: kr.preprocessing.sequence.pad_sequences(input1, self.config.seq_length_1),
            self.model.input_x_2: kr.preprocessing.sequence.pad_sequences(input2, self.config.seq_length_2),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return y_pred_cls[0]


