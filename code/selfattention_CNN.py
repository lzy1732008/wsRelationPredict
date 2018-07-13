#WMD中的attetion

#该模型是使用word2vec表示输入输出
# coding: utf-8

import tensorflow as tf
import math


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length_1 = 30  # 序列长度：
    seq_length_2 = 50  # 序列长度：法条
    attention_size = 5

    num_classes = 2  # 类别数


    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸

    num_layers = 2  # 隐藏层层数
    hidden_dim = 128  # 隐藏层神经元


    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 128  # 每批训练大小
    num_epochs = 200  # 总迭代轮次

    print_per_batch = 10  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x_1 = tf.placeholder(tf.float32, [None, self.config.seq_length_1, self.config.embedding_dim], name='input_x_1')
        self.input_x_2 = tf.placeholder(tf.float32, [None, self.config.seq_length_2, self.config.embedding_dim], name='input_x_2')
        self.input_y = tf.placeholder(tf.int32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        with tf.name_scope("attention"):
             with tf.variable_scope("linear"):
                 #首先做线性变换，其中对input_2做两次线性变换
                 newinput1 = tf.expand_dims(self.input_x_1, axis=3)
                 newinput2 = tf.expand_dims(self.input_x_2, axis=3)
                 w1 = tf.Variable(tf.random_normal([2, 2, 1 , 1], stddev=0, seed=1),
                                  trainable=True, name='w1')
                 w2 = tf.Variable(tf.random_normal([2, 2, 1, 1], stddev=0, seed=2),
                                  trainable=True, name='w2')
                 w3 = tf.Variable(tf.random_normal([2, 2, 1, 1], stddev=0, seed=3),
                                  trainable=True, name='w2')
                 Q = tf.nn.conv2d(newinput1, w1, strides=[1, 1, 1, 1], padding='SAME')[:,:,:,0]
                 K = tf.nn.conv2d(newinput2, w2, strides=[1, 1, 1, 1], padding='SAME')[:,:,:,0]
                 V = tf.nn.conv2d(newinput2, w3, strides=[1, 1, 1, 1], padding='SAME')[:,:,:,0]

             with tf.variable_scope("selfattention"):
                 t1 = tf.matmul(Q,K,adjoint_b=True)/math.sqrt(self.config.embedding_dim)       #[None,seq1,dim]*[None,dim,seq2] = [None.seq1,seq2]
                 t2 = tf.nn.softmax(t1, dim=2)
                 t3 = tf.matmul(t2, V)#[None,seq1,dim]
                 gmp3 = tf.reduce_mean(t3, axis=1)


        with tf.name_scope("cnn1"):
            # CNN layer
            with tf.variable_scope("cnn-var1"):
                conv = tf.layers.conv1d(self.input_x_1, self.config.num_filters, self.config.kernel_size, name='conv')#(?,26,256)
                # global max pooling layer
                gmp1 = tf.reduce_max(conv, reduction_indices=[1], name='gmp1')#(?,256)

        with tf.name_scope("cnn2"):
            # CNN layer
            with tf.variable_scope("cnn-var2"):
                conv = tf.layers.conv1d(self.input_x_2, self.config.num_filters, self.config.kernel_size, name='conv')#(?,26,256)
                # global max pooling layer
                gmp2 = tf.reduce_max(conv, reduction_indices=[1], name='gmp2')#(?,256)

        # with tf.name_scope("cnn3"):
        #     # CNN layer
        #     with tf.variable_scope("cnn-var3"):
        #         conv = tf.layers.conv1d(t3, self.config.num_filters, self.config.kernel_size,
        #                                 name='conv')  # (?,26,256)
        #         # global max pooling layer
        #         gmp3 = tf.reduce_max(conv, reduction_indices=[1], name='gmp3')  # (?,256)

        with tf.name_scope("concat"):
            concat = tf.concat([gmp1,gmp2],1)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(concat, self.config.hidden_dim, name='fc1')#w*input+b,其中可以在此方法中指定w,b的初始值，或者通过tf.get_varable指定
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)#根据比例keep_prob输出输入数据，最终返回一个张量
            fc = tf.nn.relu(fc)#激活函数，此时fc的维度是hidden_dim

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')#将fc从[batch_size,hidden_dim]映射到[batch_size,num_class]输出
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)#对logits进行softmax操作后，做交叉墒，输出的是一个向量
            self.loss = tf.reduce_mean(cross_entropy)#将交叉熵向量求和，即可得到交叉熵
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)




        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)#由于input_y也是onehot编码，因此，调用tf.argmax(self.input_y)得到的是1所在的下表
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
