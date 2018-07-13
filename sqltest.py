import pymysql

def connectSQL():
    connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='83621363', db='law', charset='utf8mb4')

    # 通过cursor创建游标
    cursor = connection.cursor()
    return cursor
def getftnr(ftname,cursor):
    start = ftname.index('第')
    end = ftname.index('条')
    ftmc = ftname[:start].strip()
    ftnum = ftname[start:end].strip()+'条'
    print(ftmc)
    print(ftnum)
    sql = u"SELECT  Article_text FROM law_1_article WHERE DOC_NAME = '" + ftmc + "' AND ARTICLE_SEQ = '" + ftnum + "'"

    # sql = "select  from law_1_article where doc_name='"+ftmc+"' and article_seq='"+ftnum+"';"
    print(sql)
    try:
        cursor.execute(sql)
        result = cursor.fetchone()
        return result
    except:
        return ''

cur = connectSQL()
ftname = ' 中华人民共和国刑法第十八条'
print(getftnr(ftname,cur))
cur.close()