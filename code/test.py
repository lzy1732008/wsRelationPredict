from code.predict import RnnModel
from code.predictJF import CnnModel
import jieba.posseg as pos
from code.preprocess import preprocess
import os
from flask import Flask
from flask import request
import json
import pymysql
import logging



def connectSQL():
    connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='laws', db='laws',
                                 charset='utf8mb4')

    # 通过cursor创建游标
    cursor = connection.cursor()
    return cursor


def readlines(filepath):
    with open(filepath, 'r', encoding='UTF-8') as f:
        lines = f.read().split('\n')
        return list(filter(lambda x: x.strip() != '', lines))


def precess(str1, stp, p, seq_lenght):
    words = pos.cut(str1)
    sls = []
    for word, cx in words:
        if cx == 'n' or cx == 'v' or cx == 'a':
            if word in stp:
                pass
            else:
                sls.append(word)
    input1 = [p.fixedvec([p.vector(ss) for ss in sls], seq_lenght)]
    return input1


def getftnr(ftname, cursor):
    start = ftname.index('第')
    end = ftname.index('条')
    ftmc = ftname[:start].strip()
    ftnum = ftname[start:end].strip() + '条'
    sql = "select * from law_1_article where doc_name='" + ftmc + "' and article_seq='" + ftnum + "';"
    try:
        cursor.execute(sql)
        result = cursor.fetchone()
        return result
    except:
        return ''


def predict(model1, model2, stp1, stp2, p, data_json, cursor):
    # 获取ss,jl以及法条内容列表
    ssls = data_json["ss"]
    ftmcls = data_json["ftmc"]
    jlls = data_json["jl"]
    ftls = []
    for ft in ftmcls:
        ftls.append(getftnr(ft, cursor))
    dict_re = []

    for ss in ssls:
        input1 = precess(ss, stp1, p, 30)
        for ft, ftmc in zip(ftls, ftmcls):
            input2 = precess(ft, stp1, p, 50)
            # re = model1.predict(input1, input2)
            # if re == 1:
            #     dict_re.append(ss + ":" + ftmc)

    dict_re.append('split')

    for i in range(len(ftmcls)):
        ft = str(ftls[i])
        ftmc = str(ftmcls[i])
        for jl in jlls:
            input1 = precess(jl, stp2, p, 30)
            input2 = precess(ft, stp2, p, 50)
            re = model2.predict(input1, input2)
            if re == 1:
                dict_re.append(jl + ":" + ftmc)

    return dict_re


def initnode(nodels, type, basenum, dataflag):
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
        if type == 0:
            dict_i_l['topic'] = content["name"]
            dict_i_l['detail'] = content["content"]
        else:
            dict_i_l['topic'] = gettopic(content, type, nodeindex)
            dict_i_l['detail'] = content
        dict_i_l['type'] = type
        dict_i_l['x'] = 80
        dict_i_l['y'] = 150
        relation.append(dict_i_l)
    return relation


def gettopic(s, type, index):
    if len(s) < 5:
        return s
    if type == 1:
        return "事实" + str(index)
    elif type == 2:
        return '法条' + str(index)
    elif type == 3:
        return '结论' + str(index)
    else:
        return s


def getevidencelist(data_json):
    els = []
    factls = data_json["zz"]["factList"]
    if len(factls) > 0:
        evidenceList = factls["evidenceList"]
    return els


def outputjson(dict_re, data_json):
    relation = []
    zjls = getevidencelist(data_json)
    ssls = data_json["ss"]
    ftls = data_json["ftmc"]
    jlls = data_json["jl"]
    # init flag array
    zjflag, ssflag, ftflag = [0] * len(zjls), [0] * len(ssls), [0] * len(ftls)
    basezj = 0
    basess = basezj + len(zjls)
    baseft = basess + len(ssls)
    basejl = baseft + len(ftls)

    # create zjss link
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
        ssindex = basess + ssls.index(ss)
        for e in els:
            zjflag[e] = 1
            name = e["name"]
            content = e["content"]
            dict_i_s = {}
            dict_i_s['id'] = basezj + e
            dict_i_s['caseID'] = '41722'
            dict_i_s['nodeID'] = basezj + e
            dict_i_s['parentNodeID'] = ssindex
            dict_i_s['topic'] = name
            dict_i_s['detail'] = content
            dict_i_s['type'] = 0
            dict_i_s['x'] = 80
            dict_i_s['y'] = 150
            relation.append(dict_i_s)

    for jl in jlls:
        jlindex = basejl + jlls.index(jl)
        dict_i_l = {}
        dict_i_l['id'] = jlindex
        dict_i_l['caseID'] = '41722'
        dict_i_l['nodeID'] = jlindex
        dict_i_l['parentNodeID'] = -1
        dict_i_l['topic'] = gettopic(jl, 3, jlindex)
        dict_i_l['detail'] = jl
        dict_i_l['type'] = 3
        dict_i_l['x'] = 80
        dict_i_l['y'] = 150
        relation.append(dict_i_l)
        # ft
        for ft in ftls:
            ftjl = jl + ':' + ft
            if ftjl in dict_re:
                ftflag[ftls.index(ft)] = 1
                ftindex = baseft + ftls.index(ft)
                dict_i_j = {}
                dict_i_j['id'] = ftindex
                dict_i_j['caseID'] = '41722'
                dict_i_j['nodeID'] = ftindex
                dict_i_j['parentNodeID'] = jlindex
                dict_i_j['topic'] = gettopic(ft, 2, ftindex)
                dict_i_j['detail'] = ft
                dict_i_j['type'] = 2
                dict_i_j['x'] = 80
                dict_i_j['y'] = 150
                relation.append(dict_i_j)
                # ss
                for ss in ssls:
                    ssft = ss + ":" + ft
                    if ssft in dict_re:
                        ssflag[ssls.index(ss)] = 1
                        ssindex = basess + ssls.index(ss)
                        dict_i_z = {}
                        dict_i_z['id'] = ssindex
                        dict_i_z['caseID'] = '41722'
                        dict_i_z['nodeID'] = ssindex
                        dict_i_z['parentNodeID'] = jlindex
                        dict_i_z['topic'] = gettopic(ss, 1, ssindex)
                        dict_i_z['detail'] = ss
                        dict_i_z['type'] = 1
                        dict_i_z['x'] = 80
                        dict_i_z['y'] = 150
                        relation.append(dict_i_z)
    # check all ss\ft\zj which has no link,and set node for them
    r1 = initnode(nodels=zjls, type=0, basenum=basezj, dataflag=zjflag)
    r2 = initnode(nodels=ssls, type=1, basenum=basess, dataflag=ssflag)
    r3 = initnode(nodels=ftls, type=2, basenum=baseft, dataflag=ftflag)
    relation.extend(r1)
    relation.extend(r2)
    relation.extend(r3)
    return json.dumps(relation)



def getRelation():
    requestJson = {
  "ss":["撒地方","sdhf"],
  "ftmc":["中华人民共和国刑法第一百三十三条"],
  "jl":["刑法 条 拘役 缓刑 期限 不能 少于 缓刑 期限 不能 少于 缓刑 期限 判决 确定 计算","谅解 认罪 悔罪 具体 情节 刑法 条 规定 判决 犯 交通 肇事罪 判处 缓刑"]
   }


    savedict_ss = '../source/事实到法条/checkpoints/textlstm-cnn-att1'
    savedict_jl = '../source/法条到结论/checkpoints/textselfattention_cnn_0711_64bit'
    save_path_ss = os.path.join(savedict_ss, 'best_validation')
    save_path_jl = os.path.join(savedict_jl, 'best_validation')
    word2vecpath = '../source/2014model_size64.model'
    stopwords = readlines('../source/stopwords.txt')
    stopwordss = readlines('../source/num<20-ss.txt')
    stopwordjl = readlines('../source/num<20-jl.txt')
    stp1 = stopwords.extend(stopwordss)
    stp2 = stopwords.extend(stopwordjl)
    model1 = RnnModel(save_path_ss,0)
    model1 = ''
    model2 = CnnModel(save_path_jl)
    p = preprocess(word2vecpath)
    p.load_models()
    cursor = connectSQL()
    dict_re = predict(model1, model2, stp1, stp2, p, requestJson, cursor)
    print(dict_re)





# requestJson = {
#   "ss":["上的粉红色"],
#   "ftmc":["中华人民共和国刑法第一百三十三条"],
#   "jl":["刑法 条 拘役 缓刑 期限 不能 少于 缓刑 期限 不能 少于 缓刑 期限 判决 确定 计算",""]
# }


# word2vecpath = '../source/2014model_size64.model'
# p = preprocess(word2vecpath)
# p.load_models()
# ssls = requestJson["jl"]
# for ss in ssls:
#     str1 = ss
#     words = pos.cut(str1)
#     sls = []
#     stp = []
#     for word, cx in words:
#         if cx == 'n' or cx == 'v' or cx == 'a':
#             if word in stp:
#                 pass
#             else:
#                 sls.append(word)
# input1 = [p.fixedvec([p.vector(ss) for ss in sls], 30)]
# print(input1)







