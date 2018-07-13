from code.predictJF import CnnModel
from code.predictSF import RnnModel
import jieba.posseg as pos
from code.preprocess import preprocess
import os
from flask import Flask
from flask import request
import json
import pymysql
import logging
import tensorflow as tf
import re


app = Flask(__name__)

def connectSQL():
    connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='83621363', db='law',
                                 charset='utf8mb4')

    # 通过cursor创建游标
    cursor = connection.cursor()
    return cursor

def readlines(filepath):
    with open(filepath,'r',encoding='UTF-8') as f:
        lines = f.read().split('\n')
        return list(filter(lambda x:x.strip()!='',lines))


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


def getftnr(ftname,cursor):
    #remove <>/<<>>

    ftname = ftname.replace('《','')
    ftname = ftname.replace('》', '')
    ftname = ftname.replace('（', '')
    ftname = ftname.replace('）', '')
    ftname = ftname.replace('(','')
    ftname = ftname.replace(')','')
    ftname = ftname.replace('<', '')
    ftname = ftname.replace('>', '')


    start = ftname.index('第')
    end = ftname.index('条')
    ftmc = ftname[:start].strip()
    ftnum = ftname[start:end].strip()+'条'

    sql =  u"SELECT  Article_text FROM law_1_article WHERE DOC_NAME = '" + ftmc + "' AND ARTICLE_SEQ = '" + ftnum + "'"
    try:
        cursor.execute(sql)
        result = cursor.fetchone()
        return result[0]
    except:
        return ''

def predictSFR(model,stp,p,data_json,cursor):
    ssls = data_json["ss"]
    ftmcls = data_json["ftmc"]
    jlls = data_json["jl"]
    ftls = []
    for ft in ftmcls:
        ftnr = getftnr(ft.strip(), cursor)
        ftls.append(ftnr)
    dict_re = []

    for ss in ssls:
        for ft, ftmc in zip(ftls, ftmcls):
            input1 = precess(ss, stp, p, 30)
            input2 = precess(ft, stp, p, 50)
            re = model.predict(input1, input2)
            print(ss,ft,re)
            if re == 1:
                dict_re.append(ss + ":" + ftmc)
    return dict_re

def predictJFR(model,stp,p,data_json,cursor,dict_re):
    dict_re.append('split')
    # 获取ss,jl以及法条内容列表
    ssls = data_json["ss"]
    ftmcls = data_json["ftmc"]
    jlls = data_json["jl"]
    ftls = []
    for ft in ftmcls:
        ftnr = getftnr(ft.strip(), cursor)
        ftls.append(ftnr)
    for ft, ftmc in zip(ftls, ftmcls):
        for jl in jlls:
            input1 = precess(jl, stp, p, 30)
            input2 = precess(ft, stp, p, 50)
            re = model.predict(input1, input2)
            if re == 1:
                dict_re.append(jl + ":" + ftmc)
            print(ft,jl,re)
    return dict_re


def predict(model1,model2,stp1,stp2,p,data_json,cursor):
    #获取ss,jl以及法条内容列表
    ssls = data_json["ss"]
    ftmcls = data_json["ftmc"]
    jlls = data_json["jl"]
    ftls = []
    for ft in ftmcls:
        ftls.append(getftnr(ft,cursor))
    dict_re = []

    for ss in ssls:
        for ft,ftmc in zip(ftls,ftmcls):
            input1 = precess(ss,stp1,p,30)
            input2 = precess(ft,stp1,p,50)
            re = model1.predict(input1,input2)
            if re == 1:
                dict_re.append(ss+":"+ftmc)

    dict_re.append('split')

    for ft,ftmc in zip(ftls,ftmcls):
        for jl in jlls:
            input1 = precess(jl, stp2, p, 30)
            input2 = precess(ft, stp2, p, 50)
            re = model2.predict(input1, input2)
            if re == 1:
                dict_re.append(jl + ":" + ftmc)

    return dict_re

def initnode(nodels,type,basenum,dataflag):
    relation = []
    for content in nodels:
        if dataflag[nodels.index(content)] == 1:
           continue
    
        nodeindex = basenum + nodels.index(content)
        dict_i_l = {}
        dict_i_l['id'] = nodeindex
        dict_i_l['caseID'] = '41722'
        dict_i_l['nodeID'] = nodeindex
        dict_i_l['parentNodeID'] = -1
        if type==0:
           dict_i_l['topic'] = content["name"]
           dict_i_l['detail'] = content["content"]
        else:
           dict_i_l['topic'] = gettopic(content,type,nodeindex)
           dict_i_l['detail'] = content
        dict_i_l['type'] = type
        dict_i_l['x'] = 80
        dict_i_l['y'] = 150
        relation.append(dict_i_l)
    return relation
    

def gettopic(s,type,index):
    if len(s)<5:
        return s
    if type==1:
        return "事实"+str(index)
    elif type==2:
        return '法条'+ str(index)
    elif type==3:
        return '结论' + str(index)
    else:
        return s
def getevidencelist(data_json):
    els = []
    factls = data_json["zz"]["factList"]
    if len(factls)>0:
        els = factls[0]["evidenceList"]
    return els
           
def outputjson(dict_re,data_json):
    relation = []
    zjls = getevidencelist(data_json)
    ssls = data_json["ss"]
    ftls = data_json["ftmc"]
    jlls = data_json["jl"]

    #init flag array
    zjflag,ssflag,ftflag = [0]*len(zjls), [0]*len(ssls), [0]*len(ftls)
    basezj = 0 
    basess = basezj+len(zjls)
    baseft = basess+len(ssls)
    basejl = baseft+len(ftls)
 
    
    #create zjss link
    factls = data_json["zz"]["factList"]
    for line in factls:
        ss = line["content"]
        linklist = line["linkPointList"]
        els = []
        for link in linklist:
            eindex = link["index"]
            if eindex in els:
               pass
            else:
               els.append(eindex)
        ssindex = basess + ssls.index(ss.strip())
        for e in els:
           zjflag[e]=1
           name = zjls[e]["name"]
           content = zjls[e]["content"]
           dict_i_s = {}
           dict_i_s['id'] = basezj+e
           dict_i_s['caseID'] = '41722'
           dict_i_s['nodeID'] = basezj+e
           dict_i_s['parentNodeID'] = ssindex
           dict_i_s['topic'] = name
           dict_i_s['detail'] = content
           dict_i_s['type'] = 0
           dict_i_s['x'] = 80
           dict_i_s['y'] = 150
           relation.append(dict_i_s)

    for jl in jlls:
        jlindex = basejl + jlls.index(jl.strip())
        dict_i_l = {}
        dict_i_l['id'] = jlindex
        dict_i_l['caseID'] = '41722'
        dict_i_l['nodeID'] = jlindex
        dict_i_l['parentNodeID'] = -1
        dict_i_l['topic'] = gettopic(jl,3,jlindex)
        dict_i_l['detail'] = jl
        dict_i_l['type'] = 3
        dict_i_l['x'] = 80
        dict_i_l['y'] = 150
        relation.append(dict_i_l)
        #ft
        for ft in ftls:
            ftjl = jl+':'+ft
            if ftjl in dict_re:
                ftflag[ftls.index(ft)] = 1
                ftindex = baseft + ftls.index(ft)
                dict_i_j = {}
                dict_i_j['id'] = ftindex
                dict_i_j['caseID'] = '41722'
                dict_i_j['nodeID'] = ftindex
                dict_i_j['parentNodeID'] = jlindex
                dict_i_j['topic'] = gettopic(ft,2,ftindex)
                dict_i_j['detail'] = ft
                dict_i_j['type'] = 2
                dict_i_j['x'] = 80
                dict_i_j['y'] = 150
                relation.append(dict_i_j)
                #ss
                for ss in ssls:
                    ssft = ss+":"+ft
                    if ssft in dict_re:
                        ssflag[ssls.index(ss)] = 1
                        ssindex = basess + ssls.index(ss)
                        dict_i_z = {}
                        dict_i_z['id'] = ssindex
                        dict_i_z['caseID'] = '41722'
                        dict_i_z['nodeID'] = ssindex
                        dict_i_z['parentNodeID'] = jlindex
                        dict_i_z['topic'] = gettopic(ss,1,ssindex)
                        dict_i_z['detail'] = ss
                        dict_i_z['type'] = 1
                        dict_i_z['x'] = 80
                        dict_i_z['y'] = 150
                        relation.append(dict_i_z)
    #check all ss\ft\zj which has no link,and set node for them 
    r1 = initnode(nodels=zjls,type=0,basenum=basezj,dataflag=zjflag)
    r2 = initnode(nodels=ssls,type=1,basenum=basess,dataflag=ssflag)
    r3 = initnode(nodels=ftls,type=2,basenum=baseft,dataflag=ftflag)
    relation.extend(r1)
    relation.extend(r2)
    relation.extend(r3)
    return json.dumps(relation)

@app.route('/getRelation',methods=['POST'])
def getRelation():
    logging.basicConfig(filename='logger1.log', level=logging.INFO)
    data = request.get_data()
    requestJson = json.loads(data.decode('UTF-8'))
    logging.info('get data:')
    logging.info(requestJson)
############修改
    # with open('../source/ws.jsonon', 'r', encoding='utf-8') as f:
    #     requestJson = json.load(f)
############修改

    savedict_ss = '../source/事实到法条/checkpoints/textlstm-cnn-att1'
    savedict_jl = '../source/法条到结论/checkpoints/textselfattention_cnn_0711_64bit'
    save_path_ss = os.path.join(savedict_ss, 'best_validation')
    save_path_jl = os.path.join(savedict_jl, 'best_validation')
    word2vecpath = '../source/2014model_size64.model'
    stopwords = readlines('../source/stopwords.txt')
    stp1 = readlines('../source/num<20-ss.txt')
    stp2 = readlines('../source/num<20-jl.txt')
    stp1.extend(stopwords)
    stp2.extend(stopwords)
    p = preprocess(word2vecpath)
    p.load_models()
    cursor = connectSQL()
    #预测事实法条
    g1 = tf.Graph()
    isess1 = tf.Session(graph=g1)
    with g1.as_default():
        model1 = RnnModel(save_path_ss,isess1)
        dict_re = predictSFR(model1, stp1, p, requestJson, cursor)


    #预测法条结论
    g2 = tf.Graph()
    isess2 = tf.Session(graph=g2)
    with g2.as_default():
        model2 = CnnModel(save_path_jl,isess2)
        dict_re = predictJFR(model2, stp2, p, requestJson, cursor, dict_re)


    print(dict_re)
    opjs = outputjson(dict_re,requestJson)


    logging.info(json.loads(opjs))
    # dict_re format:['ss1:ftmc1','ss2:ftmc2','split','jl1:ftmc1','jl2:ftmc2']  is a list type
    cursor.close()
    return opjs


#
if __name__=='__main__':
    app.run(host='localhost',port=5001)









