
#/usr/bin/python
# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pyspark import SparkConf, SparkContext
from pyspark.sql import HiveContext
import pymysql.cursors
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import os
#os.environ["PYSPARK_PYTHON"]="/usr/bin/python2"

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

#链接hive时的IP和端口号；如果没有相关IP和端口，暂时不设置：     
#conf = (SparkConf().setMaster("spark://192.168.12.67:7077").setAppName("My app").set("spark.executor.memory", "1g"))
conf = (SparkConf().setAppName("My app").set("spark.executor.memory", "3g"))
sc = SparkContext(conf = conf)
sqlContext = HiveContext(sc)
my_dataframe = sqlContext.sql("Select * from bi1.result_name2")


##将数据写入mysql的数据库，但需要先通过sqlalchemy.create_engine建立连接,且字符编码设置为utf8，否则有些latin字符不能处理  
yconnect = create_engine('mysql+mysqldb://root:123456@192.168.12.41:3306/ezp-opt?charset=utf8')
Session = sessionmaker(bind=yconnect)
session = Session()

#全表注册为临时表总体应用:
my_dataframe.registerTempTable('tmpTable')

#(一)针对积分兑券的业务分类及排序
vucsm =  sqlContext.sql("select sum(count) from tmpTable where actType = 1")
vucsm_sum = round(vucsm.toPandas().ix[0,0],30)
voucher = sqlContext.sql("select name,vipId,brandId,copId,sum(count) as freq from tmpTable where actType = 1 group by name,vipId,brandId,copId")
voucher_df = voucher.toPandas()
try:

    ColId = voucher_df
    
    #插入mysql数据的权重    
    poiu = session.execute('select Weight from opt_bi_dynamiclable_cfg where `Subject` = 2 and `Type` = 1')
    weightone = 0
    for row in poiu:
        weightone = float(row[0])
    voucher_df['fraction'] = voucher_df['freq'].map(lambda x: int((x/vucsm_sum)*1000000*weightone))
    
    #算法分类计算区间数值
    k = 5     
    iteration = 500
    data = voucher_df.drop(columns=['name','vipId','brandId','copId'])
    ##n_jobs是并行数，一般等于CPU数较好
    kmodel = KMeans(n_clusters = k, n_jobs = 4)
    kmodel.fit(data)
    r1 = pd.Series(kmodel.labels_).value_counts()
    r2 = pd.DataFrame(kmodel.cluster_centers_)
    r = pd.concat([r2, r1], axis = 1)
    
    r.columns = list(data.columns) + [u'num_categories']
    #针对聚类求得分割区间点在lihudong列表中：
    lihudong = r['fraction'].values.tolist()
    lihudong.sort()
    r = pd.concat([data, pd.Series(kmodel.labels_, index = data.index)], axis =1)
    r.columns = list(data.columns) + [u'clu_categoryone']
    
    datanote = voucher_df.drop(columns=['name','vipId','brandId','copId','freq'])
    dalist = datanote['fraction'].values.tolist()
    damax = max(dalist)
    damin = min(dalist)
    #求解分类的类型分为三类：1、最低类；2、中等类；3、最高类；
    
    def fun_test(x):
        if x <= lihudong[2]:
            return 1
        elif x >=  lihudong[4]:
            return 3
        else:
            return 2
    #rclu_categoryone
    r['clu_category'] = r.apply(lambda row: fun_test(row['fraction']), axis=1)
    del r['clu_categoryone']
    del r['freq']
    #求解每一类的人数
    onenum = np.sum(list(map(lambda x: x == 1, r['clu_category'])))
    twonum = np.sum(list(map(lambda x: x == 2, r['clu_category'])))
    threenum = np.sum(list(map(lambda x: x == 3, r['clu_category'])))
    r['classification'] = r.clu_category.apply(lambda x: 2)
    del ColId['fraction']
    result = pd.concat([ColId, r], axis=1)
    result.to_csv('/home/jifenduiquan.csv')
    print(result.head(1))
    
    #向数据库mysql,ezp-opt.opt_bi_range插入数据
    rangstart = "%d-%d,%d;%d-%d,%d;%d-%d,%d"%(damin,lihudong[2],onenum,lihudong[2],lihudong[4],twonum,lihudong[4],damax,threenum)
    nest_dict={'id':{1:1},'ActType':{1:2},'BrandId':{1:61},'ShardingId':{1:47},'Rang':{1:rangstart},'maximum':{1:damax}}
    
    cols=['Id','ActType','BrandId','ShardingId','Rang','maximum']
    df=pd.DataFrame(nest_dict)
    df=df.ix[:,cols]
    df.ix[:,cols]
    l = 0
    r = 7
    length = len(df)
    while(l<length):
        pd.io.sql.to_sql(df[l:r],'opt_bi_range',yconnect, schema='ezp-opt',if_exists='append',index=False)
        l+=1
        r+=1
        
    #数据写入hdfs中
    from hdfs import InsecureClient
    hdfs_client = InsecureClient('http://cluster2-master:50070', user='root')
    if len(result.head(1)):
        pandas_df = result
        pandas_df.to_csv('/home/jifenduiquan.csv')
        dir_path = '/tmp'
        file_list = hdfs_client.list(dir_path)
        target_file = 'jifenduiquan.csv'
        append = True if target_file in file_list else False
        with hdfs_client.write("{}/{}".format(dir_path, target_file), append=append, encoding='utf-8') as writer:
            pandas_df.to_csv(writer,header=True)    
    
except:
    pass
    
    
    
#(二)游戏one
vutow =  sqlContext.sql("select * from tmpTable where actType = 2 or actType =3 or actType =4 or actType =5 or actType =6 or actType = 0")
vouchertow_hu = vutow.toPandas()
try:

    hdyxone = session.execute('select Weight from opt_bi_dynamiclable_cfg where `Subject` = 4 and `Type` = 1')
    weighthone = 0
    for one in hdyxone:
        weighthone = float(one[0])    
    hdyxtwo = session.execute('select Weight from opt_bi_dynamiclable_cfg where `Subject` = 4 and `Type` = 2')
    weighthtwo = 0
    for two in hdyxtwo:
        weighthtwo = float(two[0])    
    hdyxthree = session.execute('select Weight from opt_bi_dynamiclable_cfg where `Subject` = 4 and `Type` = 3')
    weighththree= 0
    for three in hdyxthree:
        weighththree = float(three[0])    
    hdyxfoure = session.execute('select Weight from opt_bi_dynamiclable_cfg where `Subject` = 4 and `Type` = 4')
    weighthfoure = 0
    for foure in hdyxfoure:
        weighthfoure = float(foure[0])    
    hdyxfive = session.execute('select Weight from opt_bi_dynamiclable_cfg where `Subject` = 4 and `Type` = 5')
    weighthfive = 0
    for five in hdyxfive:
        wweighthfive = float(five[0])
    
    def hd_test(x,y):
        if x == 0:
            return y*(1.000000)
        elif x == 2:
            return y*wweighthfive
        elif x == 3:
            return y*weighthfoure
        elif x == 4:
            return y*weighthone
        elif x == 5:
            return y*weighthtwo
        elif x == 6:
            return y*weighththree
        else:
            pass
    vouchertow_hu['value'] = vouchertow_hu.apply(lambda row: hd_test(row['acttype'], row['count']), axis=1)
    schemacc1=sqlContext.createDataFrame(vouchertow_hu)
    schemacc1.registerTempTable("hudong")
    qiuhe =  sqlContext.sql("select sum(value) from hudong where actType = 2 or actType =3 or actType =4 or actType =5 or actType =6 or actType = 0")
    qiuhetow_sum = round(qiuhe.toPandas().ix[0,0],30)
    
    hdyx = sqlContext.sql("select name,vipId,brandId,copId,sum(value) as freq from hudong group by name,vipId,brandId,copId")
    auyt_dfhd = hdyx.toPandas()
    print(auyt_dfhd.head(1))
    ColIdhd = auyt_dfhd
    
    auyt_dfhd['fraction'] = auyt_dfhd['freq'].map(lambda x: int((x/qiuhetow_sum)*10000000))
    k = 5     
    iteration = 30
    datahd = auyt_dfhd.drop(columns=['name','vipId','brandId','copId'])
    #n_jobs是并行数，一般等于CPU数较好
    kmodelhd = KMeans(n_clusters = k, n_jobs = 4)
    kmodelhd.fit(datahd)
    r1hd = pd.Series(kmodelhd.labels_).value_counts()
    r2hd = pd.DataFrame(kmodelhd.cluster_centers_)
    rqh = pd.concat([r2hd, r1hd], axis = 1)
    rqh.columns = list(datahd.columns) + [u'num_categories']
    
    
    #针对聚类求得分割区间点在lihudong列表中：
    lihudong = rqh['fraction'].values.tolist()
    lihudong.sort()
    rqh = pd.concat([datahd, pd.Series(kmodelhd.labels_, index = datahd.index)], axis =1)
    rqh.columns = list(datahd.columns) + [u'clu_categoryone']
    
    datahdnotehd = auyt_dfhd.drop(columns=['name','vipId','brandId','copId','freq'])
    dalisthd = datahdnotehd['fraction'].values.tolist()
    damaxhd = max(dalisthd)
    daminhd = min(dalisthd)
    
    #求解分类的类型分为三类：1、最低类；2、中等类；3、最高类；
    def hudong_test(x):
        if x <= lihudong[2]:
            return 1
        elif x >=  lihudong[4]:
            return 3
        else:
            return 2
    #rclu_categoryone
    rqh['clu_category'] = rqh.apply(lambda row: hudong_test(row['fraction']), axis=1)
    del rqh['clu_categoryone']
    del rqh['freq']
    #求解每一类的人数
    onehd = np.sum(list(map(lambda x: x == 1, rqh['clu_category'])))
    twohd = np.sum(list(map(lambda x: x == 2, rqh['clu_category'])))
    threehd = np.sum(list(map(lambda x: x == 3, rqh['clu_category'])))
    rqh['classification'] = rqh.clu_category.apply(lambda x: 4)
    del ColIdhd['fraction']
    resulthd = pd.concat([ColIdhd, rqh], axis=1)
    resulthd.to_csv('/home/hudongyouxi.csv')
    print(resulthd.head(1))
    
    
    #向数据库mysql,ezp-opt.opt_bi_range插入数据
    hdstart = "%d-%d,%d;%d-%d,%d;%d-%d,%d"%(daminhd,lihudong[2],onehd,lihudong[2],lihudong[4],twohd,lihudong[4],damaxhd,threehd)
    hddict={'id':{1:1},'ActType':{1:4},'BrandId':{1:61},'ShardingId':{1:47},'Rang':{1:hdstart},'maximum':{1:damaxhd}}
    
    colshd=['Id','ActType','BrandId','ShardingId','Rang','maximum']
    dfhd=pd.DataFrame(hddict)
    dfhd=dfhd.ix[:,colshd]
    dfhd.ix[:,colshd]
    lhd = 0
    rhd = 7
    length = len(dfhd)
    while(lhd<length):
        pd.io.sql.to_sql(dfhd[lhd:rhd],'opt_bi_range',yconnect, schema='ezp-opt',if_exists='append',index=False)
        lhd+=1
        rhd+=1
    print(dfhd)
    
    #数据写入hdfhds中
    from hdfs import InsecureClient
    #hdfs_client = InsecureClient('http://cluster2hd-master:50070', user='root')
    if len(resulthd.head(1)):
        panhd = resulthd
        panhd.to_csv('/home/hudongyouxi.csv')
        hddir_path = '/tmp'
        hdle_list = hdfs_client.list(hddir_path)
        tahd_file = 'hudongyouxi.csv'
        appendhd = True if tahd_file in hdle_list else False
        with hdfs_client.write("{}/{}".format(hddir_path, tahd_file), append=appendhd, encoding='utf-8') as writer:
            panhd.to_csv(writer,header=True)
            
except:
    pass        
        
        
        
#（三）完善资料分类：非常配合：一般配合：不配合
wstow =  sqlContext.sql("select * from tmpTable where actType = 7 or actType = 8")
vouwsch_ws = wstow.toPandas()
try:

    #插入mysql数据的权重
    wanshone = session.execute('select Weight from opt_bi_dynamiclable_cfg where `Subject` = 3 and `Type` = 1')
    weightws = 0
    for ws in wanshone:
        weightws = float(ws[0])
    
    def ws_test(x,y):
        if x == 7:
            return y*(1.000000)
        elif x == 8:
            return y*(1.000000)
        else:
            pass
    vouwsch_ws['value'] = vouwsch_ws.apply(lambda row: ws_test(row['acttype'], row['count']),axis=1)
    schemaws=sqlContext.createDataFrame(vouwsch_ws)
    schemaws.registerTempTable("wstp")
    wsqhe =  sqlContext.sql("select sum(value) from wstp where actType = 7 or actType = 8")
    wsqwtw_sum = round(wsqhe.toPandas().ix[0,0],30)
    print(wsqwtw_sum)
    hdyxws = sqlContext.sql("select name,vipId,brandId,copId,sum(value) as freq from wstp group by name,vipId,brandId,copId")
    auyt_ws = hdyxws.toPandas()
    print(auyt_ws.head(1))
    ColIdws = auyt_ws
    auyt_ws['fraction'] = auyt_ws['freq'].map(lambda x: int((x/wsqwtw_sum)*1000000*weightws))
    
    k = 5     
    iteration = 20
    dataws = auyt_ws.drop(columns=['name','vipId','brandId','copId'])
    #n_jobs是并行数，一般等于CPU数较好
    kmodelws = KMeans(n_clusters = k, n_jobs = 4)
    kmodelws.fit(dataws)
    r1ws = pd.Series(kmodelws.labels_).value_counts()
    r2ws = pd.DataFrame(kmodelws.cluster_centers_)
    rqhws = pd.concat([r2ws, r1ws], axis = 1)
    rqhws.columns = list(dataws.columns) + [u'num_categories']
    #针对聚类求得分割区间点在liefen列表中：
    wsfen = rqhws['fraction'].values.tolist()
    wsfen.sort()
    rqhws = pd.concat([dataws, pd.Series(kmodelws.labels_, index = dataws.index)], axis =1)
    rqhws.columns = list(dataws.columns) + [u'clu_categoryone']
    
    datawsnws = auyt_ws.drop(columns=['name','vipId','brandId','copId','freq'])
    dalistws = datawsnws['fraction'].values.tolist()
    damaxws = max(dalistws)
    daminws = min(dalistws)
    #求解分类的类型分为三类：1、最低类；2、中等类；3、最高类；
    def ziliao_test(x):
        if x <= wsfen[2]:
            return 1
        elif x >=  wsfen[4]:
            return 3
        else:
            return 2
    #rclu_categoryone
    rqhws['clu_category'] = rqhws.apply(lambda row: ziliao_test(row['fraction']), axis=1)
    del rqhws['clu_categoryone']
    del rqhws['freq']
    #求解每一类的人数
    onews= np.sum(list(map(lambda x: x == 1, rqhws['clu_category'])))
    twows = np.sum(list(map(lambda x: x == 2, rqhws['clu_category'])))
    threews = np.sum(list(map(lambda x: x == 3, rqhws['clu_category'])))
    
    rqhws['classification'] = rqhws.clu_category.apply(lambda x: 3)
    del rqhws['fraction']
    resultws = pd.concat([ColIdws, rqhws], axis=1)
    resultws.to_csv('/home/wanshanziliao.csv')
    print(resultws.head(1))
    
    
    #向数据库mysql,ezp-opt.opt_bi_range插入数据
    wsstart = "%d-%d,%d;%d-%d,%d;%d-%d,%d"%(daminws,wsfen[2],onews,wsfen[2],wsfen[4],twows,wsfen[4],damaxws,threews)
    wsdict={'id':{1:1},'ActType':{1:3},'BrandId':{1:61},'ShardingId':{1:47},'Rang':{1:wsstart},'maximum':{1:damaxws}}
    
    colsws=['Id','ActType','BrandId','ShardingId','Rang','maximum']
    dfws=pd.DataFrame(wsdict)
    dfws=dfws.ix[:,colsws]
    dfws.ix[:,colsws]
    lws = 0
    rws = 7
    length = len(dfws)
    while(lws<length):
        pd.io.sql.to_sql(dfws[lws:rws],'opt_bi_range',yconnect, schema='ezp-opt',if_exists='append',index=False)
        lws+=1
        rws+=1
    print(dfws)
    
    #数据写入hdfwss中
    from hdfs import InsecureClient
    #hdfs_client = InsecureClient('http://cluster2ws-master:50070', user='root')
    if len(resultws.head(1)):
        panws = resultws
        panws.to_csv('/home/wanshanziliao.csv')
        wsdir_path = '/tmp'
        wsle_list = hdfs_client.list(wsdir_path)
        taws_file = 'wanshanziliao.csv'
        appendws = True if taws_file in wsle_list else False
        with hdfs_client.write("{}/{}".format(wsdir_path, taws_file), append=appendws, encoding='utf-8') as writer:
            panws.to_csv(writer,header=True)
            
except:
    pass            
    
        
#（四）订单评论和邀请有礼分类：优质评论：一般评论：排序:KOL
dyfstow =  sqlContext.sql("select * from tmpTable where actType = 9 or actType = 11")
vouwsch_dyf = dyfstow.toPandas()
try:

    #插入mysql数据的权重
    dingone = session.execute('select Weight from opt_bi_dynamiclable_cfg where `Subject` = 1 and `Type` = 1')
    weightdyone = 0
    for dy in dingone:
        weightdyone = float(dy[0])
    dingtwo = session.execute('select Weight from opt_bi_dynamiclable_cfg where `Subject` = 1 and `Type` = 2')
    weightdytwo = 0
    for dyl in dingtwo:
        weightdytwo = float(dyl[0])
    
    def dyf_test(x,y):
        if x == 9:
            return round(y*weightdyone,10)
        elif x == 11:
            return round(y*weightdytwo,10)
        else:
            pass
    vouwsch_dyf['value'] = vouwsch_dyf.apply(lambda row: dyf_test(row['acttype'],row['count']), axis=1)
    schemadyf=sqlContext.createDataFrame(vouwsch_dyf)
    schemadyf.registerTempTable("dyftp")
    dyfqhe =  sqlContext.sql("select sum(value) from dyftp where actType = 9 or actType = 11")
    dyfqwtw_sum = round(dyfqhe.toPandas().ix[0,0],30)
    print(dyfqwtw_sum)
    dyfyxdyf = sqlContext.sql("select name,vipId,brandId,copId,sum(value) as freq from dyftp group by name,vipId,brandId,copId")
    auyt_dyf = dyfyxdyf.toPandas()
    print(auyt_dyf.head(1))
    
    ColIddyf = auyt_dyf
    auyt_dyf['fraction'] = auyt_dyf['freq'].map(lambda x: int((x/dyfqwtw_sum)*10000000))
    
    k = 5     
    iteration = 10
    datadyf = auyt_dyf.drop(columns=['name','vipId','brandId','copId'])
    kmodeldyf = KMeans(n_clusters = k, n_jobs = 4)
    kmodeldyf.fit(datadyf)
    r1dyf = pd.Series(kmodeldyf.labels_).value_counts()
    r2dyf = pd.DataFrame(kmodeldyf.cluster_centers_)
    rqhdyf = pd.concat([r2dyf, r1dyf], axis = 1)
    rqhdyf.columns = list(datadyf.columns) + [u'num_categories']
    dingdanpinglu = rqhdyf['fraction'].values.tolist()
    dingdanpinglu.sort()
    
    
    rqhdyf = pd.concat([datadyf, pd.Series(kmodeldyf.labels_, index = datadyf.index)], axis =1)
    rqhdyf.columns = list(datadyf.columns) + [u'clu_categoryone']
    
    dadyf = auyt_dyf.drop(columns=['name','vipId','brandId','copId','freq'])
    dalistdyf = dadyf['fraction'].values.tolist()
    damaxdyf = max(dalistdyf)
    damindyf = min(dalistdyf)
    #求解分类的类型分为三类：1、最低类；2、中等类；3、最高类；
    
    def yqdd_test(x):
        if x <= dingdanpinglu[2]:
            return 1
        elif x >=  dingdanpinglu[4]:
            return 3
        else:
            return 2
    #rclu_categoryone
    rqhdyf['clu_category'] = rqhdyf.apply(lambda row: yqdd_test(row['fraction']), axis=1)
    del rqhdyf['clu_categoryone']
    del rqhdyf['freq']
    #求解每一类的人数
    oneydq = np.sum(list(map(lambda x: x == 1, rqhdyf['clu_category'])))
    twoydq = np.sum(list(map(lambda x: x == 2, rqhdyf['clu_category'])))
    threeydq = np.sum(list(map(lambda x: x == 3, rqhdyf['clu_category'])))
     
    rqhdyf['classification'] = rqhdyf.clu_category.apply(lambda x: 1)
    del auyt_dyf['fraction']
    resultdyf = pd.concat([auyt_dyf, rqhdyf], axis=1)
    resultdyf.to_csv('/home/dingdanyaoqing.csv')
    print(resultdyf.head(1))
    
    
    #向数据库mysql,ezp-opt.opt_bi_range插入数据
    dyfstart = "%d-%d,%d;%d-%d,%d;%d-%d,%d"%(damindyf,dingdanpinglu[2],oneydq,dingdanpinglu[2],dingdanpinglu[4],twoydq,dingdanpinglu[4],damaxdyf,threeydq)
    dyfdict={'id':{1:1},'ActType':{1:1},'BrandId':{1:61},'ShardingId':{1:47},'Rang':{1:dyfstart},'maximum':{1:damaxdyf}}
    
    colsdyf=['Id','ActType','BrandId','ShardingId','Rang','maximum']
    dfdyf=pd.DataFrame(dyfdict)
    dfdyf=dfdyf.ix[:,colsdyf]
    dfdyf.ix[:,colsdyf]
    ldyf = 0
    rdyf = 6
    length = len(dfdyf)
    while(ldyf<length):
        pd.io.sql.to_sql(dfdyf[ldyf:rdyf],'opt_bi_range',yconnect, schema='ezp-opt',if_exists='append',index=False)
        ldyf+=1
        rdyf+=1
    print(dfdyf)   
    
    
    #数据写入hdfdyfs中
    from hdfs import InsecureClient
    #hdfs_client = InsecureClient('http://cluster2dyf-master:50070', user='root')
    if len(resultdyf.head(1)):
        pandyf = resultdyf
        pandyf.to_csv('/home/dingdanyaoqing.csv')
        dyfdir_path = '/tmp'
        dyfle_list = hdfs_client.list(dyfdir_path)
        tadyf_file = 'dingdanyaoqing.csv'
        appenddyf = True if tadyf_file in dyfle_list else False
        with hdfs_client.write("{}/{}".format(dyfdir_path, tadyf_file), append=appenddyf, encoding='utf-8') as writer:
            pandyf.to_csv(writer,header=True)
    
    

except:
    pass




with open("/home/douban.txt","wb") as f:
        f.write(str(vucsm_sum*100)+'===='+str(qiuhetow_sum)+'===='+str(wsqwtw_sum*100)+'===='+str(dyfqwtw_sum))
            
    
    
    
  
