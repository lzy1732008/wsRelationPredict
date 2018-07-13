# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from code.lstm_cnn_attention import TextRNN, TRNNConfig
from code.preprocess import preprocess
import jieba.posseg as pos

try:
    bool(type(unicode))
except NameError:
    unicode = str




class RnnModel:
    def __init__(self,save_path,sess):
        self.config = TRNNConfig()
        self.model = TextRNN(self.config)

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

def precess(str1,stp,p,seq_lenght):

    words = pos.cut(str1)
    sls = []
    for word, cx in words:
        if cx == 'n' or cx == 'v' or cx == 'a':
            if word in list(stp):
                pass
            else:
                sls.append(word)
    input1 = [p.fixedvec([p.vector(ss) for ss in sls], seq_lenght)]
    return input1



# if __name__ == '__main__':
#     g1 = tf.Graph()
#     isess1 = tf.Session(graph=g1)
#     with g1.as_default():
#         word2vecpath = '../source/2014model_size64.model'
#         p = preprocess(word2vecpath)
#         p.load_models()
#
#         save_dir = '../source/事实到法条/checkpoints/textlstm-cnn-att1'
#         save_path = os.path.join(save_dir, 'best_validation')
#         cnn_model = RnnModel(save_path,isess1)
#         str2 = '违反交通运输管理法规，因而发生重大事故，致人重伤、死亡或者使公私财产遭受重大损失的，处三年以下有期徒刑或者拘役;交通运输肇事后逃逸或者有其他特别恶劣情节的，处三年以上七年以下有期徒刑;因逃逸致人死亡的，处七年以上有期徒刑。'
#         str1 = '经审理查明：2014年6月6日13时许，被告人张某驾驶南京华孚巴士有限公司的苏A×××××大型普通客车沿紫薇路由南向北行驶至清流路口右转弯时，刮撞同方向被害人李某驾驶的电动自行车，造成车辆受损，李某被当场碾压致死'
#         input1 = precess(str1, [], p, 30)
#         input2 = precess(str2, [], p, 50)
#         print(cnn_model.predict(input1, input2))
