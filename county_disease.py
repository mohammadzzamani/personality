

#! /usr/bin/env python
import sys
import MySQLdb
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from happierfuntokenizing.happierfuntokenizing import Tokenizer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import cPickle as pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Util import *
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFwe
from sklearn.pipeline import Pipeline

from sklearn.svm import SVR


user= ''
password = ''
database = 'county_disease'
host = ''
msg_table = 'msgs_2011to13'
# topic_table = 'feat$cat_met_a30_2000_cp_w$'+msg_table+'$user_id$16to16$1kusers'
topic_table = 'feat$cat_met_a30_2000_cp_w$'+msg_table+'$cnty$16to16'
# topic_table = 'feat$cat_fb22_all_500t_cp_w$'+msg_table+'$user_id$16to16'
# topic_table = 'feat$cat_met_a30_2000_cp_w$'+msg_table+'$user_id$16to16'
control_table = 'topDeaths_comp_10to15'
ngrams_table = 'feat$1to3gram$'+msg_table+'$cnty$16to16$0_5$pmi3_0_updated'
nbools_table = 'feat$1to3gram$'+msg_table+'$cnty$16to1$0_5$pmi3_0_updated'
# personality_feats = [ 'big5_neu', 'big5_ope', 'big5_agr', 'big5_con', 'big5_ext',]
disease_feats = [ '01hea_aar', '02mal_aar', '03res_aar', '04acc_aar', '05cer_aar',
                 '06alz_aar', '07dia_aar', '08flu_aar', '09nep_aar', '10sui_aar',
                 '11sep_aar', '12liv_aar', '13hyp_aar', '14par_aar', '15pne_aar']
# demog_feats = ['demog_age_fixed', 'demog_gender']
ses_demog_feats = ['hsgradHC03_VC93ACS3yr$10', 'bachdegHC03_VC94ACS3yr$10' , 'edu_index'  ,
                   'logincomeHC01_VC85ACS3yr$10' , 'unemployAve_BLSLAUS$0910'	, 'femalePOP165210D$10',
                   'hispanicPOP405210D$10',  'blackPOP255210D$10', 'forgnbornHC03_VC134ACS3yr$10',
                   'county_density', 'marriedaveHC03_AC3yr$10', 'median_age']
ses_demog_feats_name = ['hsgrad', 'bachdeg', 'edu', 'income' , 'unemployment', 'female',
                        'hispanic' , 'black', 'foreign' , 'density' , 'married' , 'age' ]
# control_feats = personality_feats + demog_feats


alphas=[0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001 , 0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]

def connectToDB():
    print ('connectToDB...')
    # Create SQL engine
    myDB = URL(drivername='mysql', database=database, query={
        'read_default_file' : '/home/mzamani/.my.cnf' })
    engine = create_engine(name_or_url=myDB)
    connection = engine.connect()
    return connection

def connectMysqlDB():
    print ('connectMysqlDB...')
    conn = MySQLdb.connect(host, user, password, database)
    c = conn.cursor()
    return c


def load_data_train_test(fname):
    print('load_data_train_test...')
    data = {"test": {}, "train": {}}
    with open(fname) as f:
        for line in f:
            uid, test_train, tweet = line.strip().split("\t")
            test_train = test_train.lower()
            if uid not in data[test_train]:
                data[test_train][uid] = tweet
            else:
                data[test_train][uid] += " " + tweet

    train_tweets = []
    train_uids = []
    for uid in data["train"]:
        train_tweets.append(data["train"][uid])
        train_uids.append(uid)

    test_tweets = []
    test_uids = []
    for uid in data["test"]:
        test_tweets.append(data["test"][uid])
        test_uids.append(uid)
    return train_tweets, train_uids, test_tweets, test_uids


def load_tweets(cursor, index, topic_df):
    print('load_tweets...')

    language_df = None
    ids = '\' , \''.join(topic_df.index.values.tolist())

    sql = "select {0} , message from {1} where {0} in ( \'{2}\' )".format(index, msg_table, ids)
    query = cursor.execute(sql)
    result =  query.fetchall()
    language_df = pd.DataFrame(data = result, columns = [index , 'message'])
    print ('language_df.shape: ', language_df.shape)
    language_df = language_df.groupby(index).agg({'message':lambda x:' '.join(x)})
    # language_df = result_df if language_df is None else pd.concat([language_df, result_df])

    print ('language_df.shape: ', language_df.shape)
    return language_df


def load_ngrams(cursor, index, users=None, ngrams_table = ngrams_table, threshold = 0 ):
    print('load_ngrams...')
    sql = "select distinct( feat) from {0}".format(ngrams_table)
    query = cursor.execute(sql)
    result =  query.fetchall()
    words = []
    for res in result:
        words.append(res[0])


    if users is not None:
        ids = '\' , \''.join(users.index.astype(str).values.tolist())
        # print (users.iloc[:10].index)
        # , \'?\', \'???\', \'?????\', \'???????\', \'????\', \'??????\', \'????????\')

        sql = "select group_id , feat, value, group_norm from {0} where group_id in ( \'{1}\' ) ".format(ngrams_table, ids)
    else:
        sql = "select group_id , feat, value, group_norm from {0} ".format(ngrams_table)
    # print ('sql: ' , sql)
    query = cursor.execute(sql)
    result =  query.fetchall()
    language_df = pd.DataFrame(data = result, columns = [index , 'feat', 'value', 'group_norm'])
    # language_df.feat = language_df.feat.map(lambda x: words.index(x))


    language_df.drop_duplicates(subset=['cnty','feat'], inplace=True)

    language_df.set_index(['cnty','feat'], inplace=True)
    print ( 'duplicates: ' , language_df.index.duplicated().any())
    print (language_df[language_df.index.duplicated()])
    language_df.reset_index(inplace=True)




    if threshold > 0:
        language_df.set_index(index, inplace=True)
        language_df = language_df.loc[language_df.groupby(index)['value'].sum() > threshold]
        language_df.reset_index(inplace=True)

    language_df.set_index(['cnty','feat'], inplace=True)
    print ( 'duplicates: ' , language_df.index.duplicated().any())
    language_df.reset_index(inplace=True)


    language_df = language_df.pivot(index=index, columns='feat', values='group_norm')
    print ('language_df.shape after pivot: ' , language_df.shape)
    # print (language_df.iloc[0:2])

    return language_df

def load_topics(cursor, index, users = None, gft = 500):
    print('load_topics...')
    if users is not None:
        ids = '\' , \''.join(users.index.astype(str).values.tolist())
        sql = "select group_id , feat, group_norm from {0} where group_id in ( \'{1}\' )".format(topic_table, ids)
    else:
        sql = "select group_id , feat, group_norm from {0}".format(topic_table)
    query = cursor.execute(sql)
    result =  query.fetchall()
    topic_df = pd.DataFrame(data = result, columns = [index , 'feat', 'group_norm'])
    print ('topic_df.shape: ' , topic_df.shape)
    topic_df = topic_df.pivot(index=index, columns='feat', values='group_norm')

    print ('topic_df.shape after pivot: ' , topic_df.shape)
    return topic_df

def load_controls(cursor, index, control_feats , control_feats_name = None, topic_df = None , drop_nan = False ):
    print('load_controls...')

    if control_feats_name is None:
        control_feats_name = control_feats

    feats_str  = ' , '.join(control_feats)

    print ('feats_str: ' , feats_str)
    if topic_df is not None:
        ids = '\' , \''.join(topic_df.index.values.tolist())
        sql = "select {0} , {1} from {2} where {0} in ( \'{2}\' )".format(index, feats_str, control_table, ids)
    else:
        sql = "select {0} , {1} from {2} ".format(index, feats_str, control_table)
    query = cursor.execute(sql)
    result =  query.fetchall()
    control_df = pd.DataFrame(data = result, columns = [index] + control_feats_name)
    if drop_nan:
        control_df.dropna(axis=0, how='any', inplace=True)
    print ( control_df.shape)
    return control_df


def msg_to_user_langauge(language_df, index):
    print ('msg_to_user_langauge...')
    print ('language_df.shape: ' , language_df.shape)
    language_df = language_df.groupby(index).agg({'message':lambda x:' '.join(x)})
    print ('language_df.shape after group by: ' , language_df.shape)
    print ('language_df.columns: ' , language_df.columns)
    return language_df


def load_data( index = 'cnty'):
    print('load_data...')
    try:
        cursor = connectToDB()
    except:
        print("error while connecting to database:", sys.exc_info()[0])
        raise
    if(cursor is not None):

        demog_df = load_controls(cursor,index= index, control_feats=ses_demog_feats, control_feats_name = ses_demog_feats_name)
        labels_df = load_controls(cursor, index= index, control_feats= disease_feats)
        print ('shapes:::::::::: ', demog_df.shape , ' , ' , labels_df.shape)
        # labels_df = labels_df.iloc[:200,:]

        demog_df.set_index(index, inplace=True)
        labels_df.set_index(index, inplace=True)
        # control_df.set_index('user_id', inplace=True)

        # if True:
        #     return None, None, None, demog_df, labels_df

        topic_df = load_topics(cursor, index=index , users= labels_df)
        # topic_df = topic_df.iloc[:200,:]
        ngrams_df = load_ngrams(cursor, index=index, users=labels_df, threshold= 10000)
        nbools_df = load_ngrams(cursor, index=index, users=labels_df, ngrams_table=nbools_table)

        # ngrams_df = load_ngrams(cursor) #, users=personality_df)
        # [personality_df, ngrams_df ]= match_ids([personality_df, ngrams_df ])
        # topic_df = load_topics(cursor, users = personality_df)
        # nbools_df = load_ngrams(cursor, users=personality_df, ngrams_table=nbools_table)


    return ngrams_df, nbools_df, topic_df, demog_df, labels_df

def run_tfidf(train_tweets, test_tweets=[], pickle_name ='tfidf_vectorizer.pickle' ):
    print ('run_tfidf...')
    tokenizer = Tokenizer()
    tfidf_vectorizer = TfidfVectorizer(input="content",
                                       strip_accents="ascii",
                                       decode_error="replace",
                                       analyzer="word",
                                       tokenizer=tokenizer.tokenize,
                                       ngram_range=(1, 3),
                                       stop_words="english",
                                       max_df=0.8,
                                       min_df=0.2,
                                       use_idf=True,
                                       max_features=200000)

    print ('fit_transforming')
    train_tfidf = tfidf_vectorizer.fit_transform(train_tweets)
    print ('fit_transformed')

    # with open(pickle_name, 'wb') as fin:
    #     pickle.dump(tfidf_vectorizer, fin)

    if (len(test_tweets) != 0):
        test_tfidf = tfidf_vectorizer.transform(test_tweets)
        return train_tfidf, test_tfidf
    else:
        return train_tfidf

def run_tfidf_dataframe(data, col_name, index_name=''):
    print ('run_tfidf_dataframe...')
    tweets = data[col_name]
    print (type(tweets), ' , ', len(tweets))
    print (len(tweets[0]))
    print (tweets[0])
    print (tweets[1])
    print (data.iloc[[0,1]])
    tfidf = run_tfidf(tweets)
    print ('type(tfidf): ' , type(tfidf))
    print ('tfidf.shape: ', tfidf.shape )
    print ('data.index: ' , data.index)
    print ('tfidf: ', tfidf)
    if (len(index_name) > 0):
        data = pd.DataFrame(data = tfidf.todense(), index = data[index_name])
    else:
        data = pd.DataFrame(data = tfidf.todense(), index = data.index)
    print ( 'data.shape: ' , data.shape)
    return data




def parse_cmd_input():
    print ('parse_cmd_input...')
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", metavar="F", type=str,
                        help="tab-separated input file where each line is in format USER_ID<tab>TEST/TRAIN<tab>TWEET")
    parser.add_argument("output_file", metavar="O", type=str,
                        help="name of output file to be produced")
    parser.add_argument("-d", metavar="d", type=int, default=5, nargs="?",
                        help="number of factors to produce")

    args = parser.parse_args()

    return args.input_file, args.output_file, args.d




def k_fold(data, index, folds=10):
    print ('k_fold...')
    kf = KFold(n_splits=folds,shuffle=True, random_state=1)
    fold_number = 0
    folds = pd.DataFrame(data = data.index , columns=[index])
    folds['fold'] = 0
    folds.set_index(index, inplace=True)
    # print folds
    # print ('folds.index: ' , folds.index )
    for train_index, test_index in kf.split(folds.index):
        # test_index = [idx for idx in folds.index if idx in test_index]
        # print ('test_index: ' , test_index.shape)
        # print (test_index.tolist())
        # mask = folds.index in test_index
        folds.iloc[test_index.tolist(),folds.columns.get_loc('fold')] = fold_number
        # print (folds.iloc[test_index])
        fold_number+=1

    return folds


def transform(data, type='minmax', range=(0,1)):
    scaler = MinMaxScaler(feature_range=range) if type=='minmax' else StandardScaler()

    if type == 'standard':
        data.fillna(data.mean(), inplace=True)

    scaled = scaler.fit_transform(data.values)
    data = pd.DataFrame(data = scaled, index = data.index, columns = data.columns)

    if type == 'minmax':
        data.fillna(0, inplace=True)

    return [data, scaler]

def res_control(topic_df = None, language_df=None, demog_df=None, personality_df=None, folds = 5):
    print ('res_control...')

    # min_max transform controls
    [demog_df , demog_scaler] = transform(demog_df, type='minmax')
    [personality_df , personality_scaler] = transform(personality_df, type='minmax')


    print ('language_df.shape is: ' , topic_df.shape)
    [topic_df, demog_df, personality_df] = match_ids([topic_df, demog_df, personality_df])

    # standardize data and language
    # [language_df, language_scaler] = transform(language_df, type='standard')
    [topic_df, topic_scaler] = transform(topic_df, type='standard')


    pca = PCA(n_components=50)
    topicPCA = pca.fit_transform(topic_df)
    topic_df = pd.DataFrame(data = topicPCA, index=topic_df.index)

    # pca = PCA(n_components=50)
    # languagePCA = pca.fit_transform(language_df)
    # language_df = pd.DataFrame(data = languagePCA, index=language_df.index)


    adaptedTopic = multiply(demog_df, topic_df, output_filename = None )#'csv/multiplied_topic.csv')
    [adaptedTopic , adaptedTopic_scaler] = transform(adaptedTopic, type='standard')

    pca = PCA(n_components=50)
    adaptedTopicPCA = pca.fit_transform(adaptedTopic)
    adaptedTopic = pd.DataFrame(data = adaptedTopicPCA, index=adaptedTopic.index)


    # adaptedLang = multiply(demog_df, language_df, output_filename = 'csv/multiplied_data.csv')
    # [adaptedLang , adaptedLang_scaler] = transform(adaptedLang, type='standard')
    #
    # pca = PCA(n_components=70)
    # adaptedLangPCA = pca.fit_transform(adaptedLang)
    # adaptedLang = pd.DataFrame(data = adaptedLangPCA, index=adaptedLang.index)




    # adaptedLang.to_csv('csv/adaptedLang.csv')
    # language_df.to_csv('csv/language_pca.csv')



    foldsdf = k_fold(topic_df, folds=10)

    # inferred_presonality = personality_df
    # pd.DataFrame(data=personality_df.index.values.tolist(), columns='user_id')

    # print('personality index : ' , personality_df.index)
    # personality_df = personality_df[['big5_ext']] #, 'big5_neu']]

    inferred_presonality = None
    for col in personality_df.columns:
        print (type(personality_df[[col]]), ' col: ' , col)
        inferred_col = infer_personality(topic_df, labels=personality_df[[col]], foldsdf = foldsdf, folds=10, pre='...infered_'+col+'...', col_name = col)
        inferred_presonality = inferred_col if inferred_presonality is None else \
            pd.merge(inferred_presonality, inferred_col, left_index=True, right_index=True, how='inner')

        [inferred, reported] = match_ids([inferred_presonality, personality_df])
        evaluate(reported[col], inferred[col], store=False, pre='personalityVSinferred_'+col+'_')

        # print ( personality_df[col].corrwith(inferred_presonality[col]))
    # inferred_presonality.set_index('user_id', inplace=True)


    foldsdf = k_fold(topic_df, folds=folds)

    inferred_presonality_and_demog = pd.merge(inferred_presonality, demog_df, left_index=True, right_index=True, how='inner')

    adaptedAddedTopic = pd.merge(adaptedTopic, inferred_presonality_and_demog, left_index=True, right_index=True, how='inner')
    # adaptedTopic = pd.merge(adaptedTopic, inferred_presonality, left_index=True, right_index=True, how='inner')

    # foldsdf = k_fold(inferred_presonality_and_demog, folds=folds)

    # improved_personality = pd.DataFrame(index=personality_df.index)
    # res_personality = personality_df.subtract(inferred_presonality)

    # print ( 'presonality: ' )
    # m = personality_df.mean().values[0]
    # print ('m: ' , m)
    # m = [m for i in range(personality_df.shape[0])]
    # l = personality_df.values.tolist()
    #
    # print ( type(m) , '  ,  ', type(l))
    # print (len(m), ' , ' ,len(l))
    # evaluate( l, m, 'mean_err', store=False)
    #
    #
    #
    #
    #
    # print ( 'res_personality: ' )
    # m = res_personality.mean().values[0]
    # print ('m: ' , m)
    # m = [ m for i in range(res_personality.shape[0])]
    # l = res_personality.values.tolist()
    #
    # print ( type(m) , '  ,  ', type(l))
    # print (len(m), ' , ' ,len(l))
    # evaluate(l , m , 'mean_err_res', store=False)

    [inferred_presonality, personality_df, inferred_presonality_and_demog] = match_ids([inferred_presonality, personality_df, inferred_presonality_and_demog])

    improved_personality = {}
    for col in personality_df.columns:
        print( 'col: ' , col , '  ....  ')
        # evaluate(personality_df[col], inferred_presonality[col], store=False, pre='personalityVSinferred_'+col+'_')
        data = cv(inferred_presonality_and_demog, labels=personality_df[[col]], foldsdf= foldsdf, folds = folds, pre = 'res_personality_'+col, max_depth = 6, max_features=0.8, residuals=True, col_name=col)
        # print (range(folds))
        columns = ['fold_'+str(i) for i in range(folds)]
        print (data.shape, ' , ', len(columns))
        print( personality_df.index.shape)
        improved_personality[col] = pd.DataFrame( index=personality_df.index, data= data, columns= columns)

        print ('improved_presonality[col]: ' , improved_personality[col].shape, ' , ' , improved_personality[col].columns, ' , has Nan? ', improved_personality[col].isnull().values.any())
        # print ( personality_df[col].corrwith(improved_presonality[col]))

    # inferred_presonality = improved_presonality
    res_personality = {}
    for col , value in improved_personality.iteritems():
        res_personality[col] = pd.DataFrame(index=improved_personality[col].index)
        for fold in improved_personality[col].columns:
            res_personality[col][fold]  = personality_df[col].subtract(improved_personality[col][fold])
        print (' --------- ')
        print ( res_personality[col].shape, ' , ', personality_df.shape, ', ', improved_personality[col].shape, ' , ', inferred_presonality_and_demog.shape )
        print ( inferred_presonality.shape)
        print (res_personality[col].isnull().values.any(), ' , ', improved_personality[col].isnull().values.any(), ' , ', personality_df.isnull().values.any())
        print ( res_personality[col].iloc[0:2,:])
        print ( improved_personality[col].iloc[0:2,:])
        print ('............')

    # foldsdf = k_fold(adaptedTopic, folds=folds)
    print (adaptedTopic.shape , ' , ', topic_df.shape)
    for col in personality_df.columns:
        # print (type(res_personality[[col]]))

        # all_factors_adapted = multiply(controls=personality_df.loc[:, personality_df.columns != col], language=language_df,
        #                                output_filename = 'csv/multiplied_'+col+'_data.csv', all_df=adaptedLang)

        cv(topic_df, labels=res_personality[col], foldsdf= foldsdf, folds = folds, pre = 'topic_'+col, scaler = personality_scaler)
        cv(adaptedTopic, labels=res_personality[col], foldsdf= foldsdf, folds = folds, pre = 'age&gender_adaptedTopic_'+col, scaler = personality_scaler)
        cv(adaptedAddedTopic, labels=res_personality[col], foldsdf= foldsdf, folds = folds, pre = 'age&gender_adaptedAddedTopic_'+col, scaler = personality_scaler)


        # cv(topic_df, labels=res_personality[[col]], foldsdf= foldsdf, folds = folds, pre = 'topic_'+col, scaler = personality_scaler)
        # cv(adaptedTopic, labels=res_personality[[col]], foldsdf= foldsdf, folds = folds, pre = 'age&gender_adaptedTopic_'+col, scaler = personality_scaler)
        # cv(adaptedAddedTopic, labels=res_personality[[col]], foldsdf= foldsdf, folds = folds, pre = 'age&gender_adaptedAddedTopic_'+col, scaler = personality_scaler)



    print ('<<<<< personality inferred >>>>>')






def cross_validation(index = 'cnty', topic_df = None, ngrams_df=None, nbools_df=None, demog_df=None, personality_df=None, folds = 5):
    print ('cross_validation...')

    # min_max transform controls
    demog_df.fillna(demog_df.mean(),inplace=True)
    [demog_df , demog_scaler] = transform(demog_df, type='minmax')#, range=(1,2))
    # [personality_df , personality_scaler] = transform(personality_df, type='minmax')


    ngrams_df.fillna(0, inplace=True)
    nbools_df.fillna(0, inplace=True)
    topic_df.fillna(0, inplace=True)

    # print (ngrams_df.iloc[:10].index)
    # print (topic_df.iloc[:10].index)
    # print (demog_df.iloc[:10].index)
    # print (personality_df.iloc[:10].index)

    [ngrams_df, nbools_df, topic_df, demog_df, personality_df] = match_ids([ngrams_df, nbools_df, topic_df, demog_df, personality_df])

    print (ngrams_df.shape, ' , ', topic_df.shape, ' , ', demog_df.shape , ' , ', personality_df.shape)



    # nbools_df.fillna(0, inplace=True)

    # lang_df = pd.merge(language_df, topic_df, left_index=True, right_index=True)
    # lang_df.to_csv('csv/topic_ngrams_16k_7k.csv')
    # topic_df.to_csv('csv/topic_16k_2k.csv')
    # language_df.to_csv('csv/ngrams_16k_5k.csv')
    # demog_df.to_csv('csv/demog_16k.csv')
    # personality_df.to_csv('csv/personlity_16k.csv')

    # lang_df = language_df



    # data = pd.read_csv('csv/multiplied_data.csv')
    # data.set_index('user_id', inplace=True)
    # print (data.shape, ' , ', data.columns)

    ## language_df = pd.read_csv('csv/language.csv')
    ## language_df.set_index('user_id', inplace=True)

    # demog_df = pd.read_csv('csv/demog.csv')
    # demog_df.set_index('user_id', inplace=True)
    # print (demog_df.shape, ' , ', demog_df.columns)
    # personality_df = pd.read_csv('csv/personlity.csv')
    # personality_df.set_index('user_id', inplace=True)

    print ('language_df.shape is: ' , ngrams_df.shape)
    # print ('columns:' , data.columns[0:5], ' , ', language_df.columns[0:5], ' , ', demog_df.columns, ' , ', personality_df.columns)


    separately_adapted= []
    separately_adapted_cols = []
    for col in demog_df:
        separately_adapted.append([multiply(demog_df[[col]], ngrams_df), multiply(demog_df[[col]], nbools_df), multiply(demog_df[[col]], topic_df)])
        separately_adapted_cols = separately_adapted_cols  + [col]

    adapted_ngrams = multiply(demog_df, ngrams_df) #, output_filename = 'csv/multiplied_topic.csv')
    adapted_topics = multiply(demog_df, topic_df) #, output_filename = 'csv/multiplied_topic.csv')
    adapted_nbools = multiply(demog_df, nbools_df)

    # age_ngrams = multiply(demog_df[['demog_age_fixed']], ngrams_df, inclusive = False)
    # age_nbools = multiply(demog_df[['demog_age_fixed']], nbools_df, inclusive = False)
    # age_topics = multiply(demog_df[['demog_age_fixed']], topic_df, inclusive = False)
    #
    # gender_ngrams = multiply(demog_df[['demog_gender']], ngrams_df, inclusive = False)
    # gender_nbools= multiply(demog_df[['demog_gender']], nbools_df, inclusive = False)
    # gender_topics = multiply(demog_df[['demog_gender']], topic_df, inclusive = False)

    dim_sizes  = [ 150, 75, 75, 150, 75, 50] #, 25, 25, 25, 25, 25, 25]



    # standardize data and language
    # [language_df, language_scaler] = transform(language_df, type='standard')
    # [topic_df, topic_scaler] = transform(topic_df, type='standard')


    # pca = PCA(n_components=50)
    # topicPCA = pca.fit_transform(topic_df)
    # topic_df = pd.DataFrame(data = topicPCA, index=topic_df.index)

    # pca = PCA(n_components=50)
    # languagePCA = pca.fit_transform(language_df)
    # language_df = pd.DataFrame(data = languagePCA, index=language_df.index)

    # lang_df.fillna(lang_df.mean(), inplace=True)



    # adaptedLang = multiply(demog_df, lang_df) #, output_filename = 'csv/multiplied_topic.csv')
    # [adaptedTopic , adaptedTopic_scaler] = transform(adaptedLang, type='standard')

    # pca = PCA(n_components=1000)
    # adaptedLangPCA = pca.fit_transform(adaptedLang)
    # adaptedLang = pd.DataFrame(data = adaptedLangPCA, index=adaptedLang.index)
    #
    # pca = PCA(n_components=1000)
    # langPCA = pca.fit_transform(lang_df)
    # langPCA = pd.DataFrame(data = langPCA, index=lang_df.index)
    #
    # langPCA = pd.merge(langPCA, demog_df, how='inner', left_index=True, right_index=True)



    # adaptedLang.to_csv('csv/adaptedLang.csv')
    # language_df.to_csv('csv/language_pca.csv')


    foldsdf = k_fold(topic_df, index=index, folds=folds)

    # inferred_presonality = personality_df
    # pd.DataFrame(data=personality_df.index.values.tolist(), columns='user_id')

    langData = [ ngrams_df, nbools_df, topic_df]
    # adapted_langData = [ adapted_ngrams, adapted_nbools, adapted_topics]

    adapted = [ adapted_ngrams, adapted_nbools, adapted_topics]
    adapted_and_added = [ adapted_ngrams, adapted_nbools, adapted_topics, demog_df]

    # ngrams_demog = pd.merge(ngrams_df, demog_df, how='inner', right_index=True, left_index=True)
    added_langData = [ngrams_df, nbools_df, topic_df , demog_df]

    # age_data = [ ngrams_df, nbools_df, topic_df, age_ngrams, age_nbools, age_topics]
    #
    # gender_data = [ ngrams_df, nbools_df, topic_df,gender_ngrams, gender_nbools, gender_topics]

    # print ( ngrams_df.isnull().values.any(), ' , ', ngrams_df.shape)
    # print ( topic_df.isnull().values.any() ,  ' , ', topic_df.shape)
    # print ( adapted_topics.isnull().values.any(), ' , ', adapted_topics.shape)
    # print ( adapted_ngrams.isnull().values.any(),  ' , ', adapted_ngrams.shape)

    print ( ngrams_df.shape, ' - ' , topic_df.shape, ' - ', adapted_ngrams.shape , ' - ' , adapted_topics.shape)

    print('personality index : ' , personality_df.index)

    # groupData = [   langData, adapted,  added_langData , adapted_and_added, langData]
    # groupDataName = [  'lang', 'adapted' , 'added'  , 'adapted_and_added', 'lang_residualized']
    # dim_sizes = [[200, 150, 150], [250, 200, 200], [200,150, 150, 150],  [250, 200, 200, 200], [200, 150, 150]]


    groupData = [   langData, adapted ,added_langData, langData, adapted]
    groupDataName = [  'lang', 'adapted' , 'added', 'lang_residualized', 'adapted_residualized']
    # dim_sizes = [[(150, 60), (100, 1), (100,10)], [(200, 1), (150, 0.5), (150, 0.5)], [(150, 60), (100, 1), (100,10), (100, 60)], [(150, 60), (100, 1), (100,10)]]
    dim_sizes = [[(150, 30.0), (100, 2.0), (100,10.0)], [(200, 1.0), (150, 0.5), (150, 0.5)], [(150, 30.0), (100, 2.0), (100,10.0), (100, 60.0)], [(150, 30.0), (100, 2.0), (100,10.0)], [(200, 1.0), (150, 0.5), (150, 0.5)]]

    added = langData + [demog_df]
    added_adapted = adapted + [demog_df]

    groupData = [ langData, adapted, added, added_adapted]
    groupDataName = [ 'res', 'adapted_res' , 'added_res', 'added_adapted_res' ]
    # dim_sizes = [[(150, 30.0), (100, 2.0), (200,20.0)], [(200, 1.0), (100, 0.5), (200, 1.0)] ,[(150, 30.0), (100, 2.0), (200,20.0), (200, 100)],  [(200, 1.0), (100, 0.5), (200, 1.0), (200, 100)] ]
    # dim_sizes = [[(40, 30.0), (20, 2.0), (60,20.0)], [(40, 1.0), (20, 0.5), (60, 1.0)] ,[(40, 30.0), (20, 2.0), (60,20.0), (60, 100)],  [(40, 1.0), (20, 0.5), (60, 1.0), (60, 100)] ]
    dim_sizes = [[(50, 30.0), (25, 2.0), (75,20.0)], [(50, 1.0), (25, 0.5), (75, 1.0)] ,[(50, 30.0), (25, 2.0), (75,20.0), (75, 100)],  [(50, 1.0), (25, 0.5), (75, 1.0), (75, 100)] ]


    # groupData = separately_adapted + [  langData ,added_langData , langData]
    # groupDataName = separately_adapted_cols +  [ 'lang' , 'added', 'lang_residualized']
    # dim_sizes = [(150, 50.0), (100, 5.0), (150,5.0), (100, 100.0) ]



    # groupData = [ [ngrams_df, nbools_df, topic_df, adapted_ngrams, adapted_nbools, adapted_topics, age_ngrams, age_nbools, age_topics, gender_ngrams, gender_nbools, gender_topics]]
    # groupDataName= [ 'all' ]
    inferred_presonality = None

    added_inferred_presonality = None
    for col in personality_df.columns:
        inferred_col = None
        ensemble_adapted = None
        personality = personality_df[[col]]
        personality.dropna(axis=0, how='any', inplace=True)
        for data_index in range(len(groupData)):
            data = groupData[data_index]
            data_name = groupDataName[data_index]
            print (col , ' , ', type(personality_df[[col]]), ' , ', data_name)

            # ESTIMATORS = [
            #         mean_est(),
            #         RidgeCV(alphas=alphas),
            #         GradientBoostingRegressor(n_estimators= 100, loss='ls', random_state=1, subsample=0.75, max_depth=5), #, min_impurity_decrease=0.05),
            #     ]

            # if data_index == len(groupData) - 1:
            #     residuals = True
            # else:
            #     residuals = False
            residuals = True
            # all_data = [data + [personality_df[[col]]] + [demog_df]]
            # print len(all_data)


            # if col == '03res_aar':
            #     print (personality)
            #     print (personality.shape)



            # if col == '03res_aar':
            #     print (personality)
            #     print (personality.shape)

            # print (foldsdf)


            matched_data = match_ids(data + [personality] + [demog_df] + [foldsdf])
            # print len(matched_data)



            foldsdf = matched_data[len(matched_data)-1]
            demog = [matched_data[len(matched_data)-2]]
            personality = matched_data[len(matched_data)-3]
            data = [ matched_data[i] for i in range(len(matched_data)-3)]

            # print (foldsdf)

            # if col == '03res_aar':
            #     # print (personality)
            #     for d in data:
            #         print (d.shape ,' , ', d.isnull().values.any())


            # print ( len(demog))
            # exit()
            # inferred = cv(data=data, controls = [demog_df], labels=personality_df[[col]], foldsdf = foldsdf, folds=folds, pre='...'+data_name+'...'+col+'...', col_name=data_name, residuals= residuals, dim_sizes = dim_sizes[data_index])
            # inferred =  cv(data=data, controls = demog, labels=personality, foldsdf = foldsdf, folds=folds, pre='...'+data_name+'...'+col+'...', col_name=data_name, residuals= residuals, dim_sizes = dim_sizes)
            inferred =  cv(data=data, controls = demog, labels=personality, foldsdf = foldsdf, folds=folds, pre='...'+data_name+'...'+col+'...', col_name=data_name, residuals= residuals, dim_sizes = dim_sizes[data_index])

            print ( 'inferred.shape....: ' , inferred.shape)
            inferred_col = inferred if inferred_col is None else \
                pd.merge(inferred_col, inferred, left_index=True, right_index=True, how='inner')
            print ( 'inferred_col.shape....: ' , inferred_col.shape)
            # [inferred, reported] = match_ids([inferred_col, personality_df[[col]]])
            # print (col, ' : ' , inferred.shape, ' , ', reported.shape)
            # evaluate(reported, inferred, store=False, pre='>>>>>ADAPTED>>>>personalityVSinferred_'+col+'_')

            # if data_index < len(groupData)-3:
            #     ensemble_adapted = inferred if ensemble_adapted is None else \
            #         pd.merge(ensemble_adapted, inferred, left_index=True, right_index=True, how='inner')


        ESTIMATORS = [
                    mean_est(),
                    RidgeCV(alphas=alphas),
                    GradientBoostingRegressor(n_estimators= 500, loss='ls', random_state=1, subsample=0.75, max_depth=8), #, min_impurity_decrease=0.05),
                ]



        inferred_col = pd.merge(inferred_col, demog_df,left_index=True, right_index=True, how='inner')
        inferred_col = [inferred_col]

        # matched_data = match_ids(data + [personality] + [demog_df] + [foldsdf])
        # foldsdf = matched_data[len(matched_data)-1]
        # demog = [matched_data[len(matched_data)-2]]
        # personality = matched_data[len(matched_data)-3]

        result_col = cv(data=inferred_col, controls = demog_df, labels=personality, foldsdf = foldsdf, folds=folds, pre='......'+col+'......', col_name=data_name,residuals=False, ESTIMATORS=ESTIMATORS)


        # result_col_ea = cv(data=[ensemble_adapted], controls = demog_df, labels=personality_df[[col]], foldsdf = foldsdf, folds=folds, pre='......'+col+'......ensemble_adapted.....', col_name=data_name,residuals=False, ESTIMATORS=ESTIMATORS)
        #
        #
        # ensemble_adapted = pd.merge(ensemble_adapted, demog_df,left_index=True, right_index=True, how='inner')
        # ensemble_adapted = [ensemble_adapted]
        # result_col_eaa = cv(data=ensemble_adapted, controls = demog_df, labels=personality_df[[col]], foldsdf = foldsdf, folds=folds, pre='......'+col+'......ensemble_adapted_added.....', col_name=data_name,residuals=False, ESTIMATORS=ESTIMATORS)




        # adapted_inferred_col = cv(data=adapted_langData, controls = demog_df, labels=personality_df[[col]], foldsdf = foldsdf, folds=folds, pre='...adapted_infered_'+col+'...')
        # adapted_inferred_presonality = adapted_inferred_col if adapted_inferred_presonality is None else \
        #     pd.merge(adapted_inferred_presonality, adapted_inferred_col, left_index=True, right_index=True, how='inner')
        # [inferred, reported] = match_ids([adapted_inferred_col, personality_df[[col]]])
        # print (col, ' : ' , inferred.shape, ' , ', reported.shape)
        # evaluate(reported, inferred, store=False, pre='>>>>>ADAPTED>>>>personalityVSinferred_'+col+'_')
        #
        #
        # # cv(data=adapted_langData, controls = demog_df, labels=personality_df[[col]], foldsdf= foldsdf, folds = folds, pre = 'infered_'+col)
        # # cv(data=adapted_langData, controls = demog_df, labels=personality_df[[col]], foldsdf = foldsdf, folds=folds, pre='...adapted_infered_'+col+'...')
        # inferred_col = cv(data=langData, controls = demog_df, labels=personality_df[[col]], foldsdf= foldsdf, folds = folds, pre = 'infered_'+col)
        # inferred_presonality = inferred_col if inferred_presonality is None else \
        #     pd.merge(inferred_presonality, inferred_col, left_index=True, right_index=True, how='inner')
        # [inferred, reported] = match_ids([inferred_col, personality_df[[col]]])
        # print (col, ' : ' ,inferred.shape, ' , ', reported.shape)
        # evaluate(reported, inferred, store=False, pre='>>>>>langPCA&demog_'+col+'_')
        #
        #
        # added_inferred_col = cv(data=added_langData, controls = demog_df, labels=personality_df[[col]], foldsdf= foldsdf, folds = folds, pre = 'infered_'+col)
        # added_inferred_presonality = added_inferred_col if added_inferred_presonality is None else \
        #     pd.merge(added_inferred_presonality, added_inferred_col, left_index=True, right_index=True, how='inner')
        # [inferred, reported] = match_ids([added_inferred_col, personality_df[[col]]])
        # print (col, ' : ' ,inferred.shape, ' , ', reported.shape)
        # evaluate(reported, inferred, store=False, pre='>>>>>langPCA&demog_'+col+'_')

    return

    # inferred_presonality.set_index('user_id', inplace=True)


    inferred_presonality_and_demog = pd.merge(inferred_presonality, demog_df, left_index=True, right_index=True, how='inner')
    improved_presonality = pd.DataFrame(index=personality_df.index)
    for col in personality_df.columns:
        improved_presonality[col] = cv(inferred_presonality_and_demog, labels=personality_df[[col]], foldsdf= foldsdf, folds = folds, pre = 'improved_personality_'+col, scaler = personality_scaler, max_depth = 5, max_features=1)

    inferred_presonality = improved_presonality


    print ('<<<<< personality inferred >>>>>')



    # data = pd.read_csv('multiplied_transformed_data.csv')
    foldsdf = k_fold(adaptedLang, folds=folds)
    print (adaptedLang.shape , ' , ', language_df.shape)
    for col in personality_df.columns:
        print (type(personality_df[[col]]))

        # all_factors_adapted = multiply(controls=personality_df.loc[:, personality_df.columns != col], language=language_df,
        #                                output_filename = 'csv/multiplied_'+col+'_data.csv', all_df=adaptedLang)
        all_factors_adapted = multiply(controls=inferred_presonality, language=language_df,
                                       output_filename = 'csv/multiplied_'+col+'_data.csv', all_df=adaptedLang)
        # data_all_factors = pd.read_csv('csv/multiplied_'+col+'_data.csv')
        # data_all_factors.set_index('user_id', inplace=True)
        all_factors_adapted.fillna(all_factors_adapted.mean(), inplace=True)
        pca = PCA(n_components=90)
        all_factors_adapted = pd.DataFrame(data = pca.fit_transform(all_factors_adapted) , index= all_factors_adapted.index)
        all_factors_adapted.to_csv(('csv/multiplied_'+col+'_data_pca.csv'))
        cv(all_factors_adapted, labels=personality_df[[col]], foldsdf= foldsdf, folds = folds,
           pre = 'age&gender&personality_adapted_'+col, scaler = personality_scaler )

        cv(language_df, labels=personality_df[[col]], foldsdf= foldsdf, folds = folds, pre = 'language_'+col, scaler = personality_scaler)
        cv(adaptedLang, labels=personality_df[[col]], foldsdf= foldsdf, folds = folds, pre = 'age&gender_adapted_'+col, scaler = personality_scaler)




def cross_validation_with_saved_data(index, language_df=None, demog_df=None, personality_df=None, folds = 10):
    print ('cross_validation...')





    # data = pd.read_csv('csv/multiplied_data.csv')
    # data.set_index('user_id', inplace=True)
    # print (data.shape, ' , ', data.columns)

    ## language_df = pd.read_csv('csv/language.csv')
    ## language_df.set_index('user_id', inplace=True)

    demog_df = pd.read_csv('csv/demog.csv')
    demog_df.set_index(index, inplace=True)
    print (demog_df.shape, ' , ', demog_df.columns)
    personality_df = pd.read_csv('csv/personlity.csv')
    personality_df.set_index(index, inplace=True)

    language_df = pd.read_csv('csv/language_pca.csv')
    language_df.set_index(index, inplace=True)
    print ('language_df.shape is: ' , language_df.shape)



    [language_df, demog_df, personality_df] = match_ids([language_df, demog_df, personality_df])
    data = multiply(demog_df, language_df, output_filename = 'csv/multiplied_data.csv')



    # remove nan
    data.fillna(data.mean(), inplace=True)
    language_df.fillna(language_df.mean(), inplace=True)


    # standardize data and language
    language_scaler = StandardScaler()
    scaled_language = language_scaler.fit_transform(language_df.values)
    language_df = pd.DataFrame(scaled_language, index=language_df.index, columns=language_df.columns)

    data_scaler = StandardScaler()
    scaled_data = data_scaler.fit_transform(data.values)
    data = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)


    # data = pd.read_csv('multiplied_transformed_data.csv')

    # randomly assign fold
    foldsdf = k_fold(data, folds=folds)
    print (data.shape , '  ,  ' , language_df.shape)


    for col in personality_df.columns:
        print (type(personality_df[[col]]))

        data_all_factors = multiply(personality_df.loc[:, personality_df.columns != col], language_df, all_df=data)
        data_all_factors.fillna(data_all_factors.mean(), inplace=True)
        # pca = PCA(n_components=500)
        # data_all_factors = pd.DataFrame(data = pca.fit_transform(data_all_factors) , index= data_all_factors.index)
        # data_all_factors.to_csv(('csv/multiplied_'+col+'_data_pca.csv'))
        cv(data_all_factors, labels=personality_df[[col]], foldsdf= foldsdf, folds = folds, pre = 'age&gender&personality_adapted_'+col)

        # cv(language_df, labels=personality_df[[col]], foldsdf= foldsdf, folds = folds, pre = 'language_'+col)
        # cv(data, labels=personality_df[[col]], foldsdf= foldsdf, folds = folds, pre = 'age&gender_adapted_'+col)



def infer_personality(data, labels, foldsdf, folds, pre, col_name= 'y'):
    print ('infer_personality...')
    # [data, labels, foldsdf] = match_ids([data, labels, foldsdf])
    # data.fillna(data.mean(), inplace=True)
    print ('data shapes: ' , data.shape, ' , ', labels.shape, ' , ', foldsdf.shape )

    ESTIMATORS = [
            mean_est(),
            RidgeCV(alphas=alphas),
            # GradientBoostingRegressor(n_estimators= 300, loss='ls', random_state=2, subsample=0.75, max_depth=6, max_features=0.75)#, min_impurity_decrease=0.025),
            # GradientBoostingRegressor(n_estimators= 300, loss='ls', random_state=2, subsample=0.75, max_depth=7, max_features=0.75, min_impurity_decrease=0.075),
    ]
    ESTIMATORS_NAME = [ 'mean' , 'ridgecv', 'gbr_ls6', 'gbr_ls7' ]
    YpredsAll = None

    Y = labels.values
    index = []
    for i in range(folds):

        # prepare train and test data
        test_ids = foldsdf[foldsdf['fold'] == i].index.tolist()
        train_ids = foldsdf[foldsdf['fold'] != i].index.tolist()

        Xtrain = data.loc[train_ids].values
        ytrain = labels.loc[train_ids].values
        ytrain = np.reshape(ytrain,(ytrain.shape[0], 1))
        Xtest = data.loc[test_ids].values
        ytest = labels.loc[test_ids].values
        ytest = np.reshape(ytest,(ytest.shape[0], 1))

        index = index + test_ids

        print ('train & test: ' , Xtrain.shape, ' , ', ytrain.shape , ' , ', Xtest.shape , ' , ', ytest.shape)

        Ypreds = None
        for j in range(len(ESTIMATORS)):
            estimator = ESTIMATORS[j]
            estimator.fit(Xtrain, ytrain)
            ypred = estimator.predict(Xtest)
            ypred = np.reshape(ypred ,newshape =(ypred.shape[0],1))
            Ypreds = stack_folds_preds(ypred, Ypreds, 'horizontal')
            evaluate(ytest, ypred, pre=pre+'_'+str(i)+'_'+ESTIMATORS_NAME[j]+'_', store=False)

        Ypreds = stack_folds_preds(ytest, Ypreds, 'horizontal')
        YpredsAll = stack_folds_preds(Ypreds, YpredsAll, 'vertical')
        print ('ypredsAll.shape: ' , YpredsAll.shape)


    for j in range(YpredsAll.shape[1]-1):
        print ('j: ' , j, ESTIMATORS_NAME[j], ' , ', YpredsAll[:,j].transpose().shape, '  , ' , Y.shape)
        evaluate(YpredsAll[:,YpredsAll.shape[1]-1].transpose(), YpredsAll[:,j].transpose(), pre=pre+'_'+ESTIMATORS_NAME[j]+'_')
        # evaluate(Y, YpredsAll[:,j].transpose(), pre=pre+'_'+ESTIMATORS_NAME[j]+'____', store=False)

    print ('length: ' , len(index) , ' , ' , YpredsAll[:,1].shape)

    result = pd.DataFrame(index=index, data= YpredsAll[:,1], columns=[col_name])

    return result




def cv(data, controls, labels, foldsdf, folds, pre, scaler=None, n_estimators = 300, subsample=0.75, max_depth=8, max_features = 0.7, residuals = False, col_name='y', ESTIMATORS = None, dim_sizes = None):
    print ('cv...')

    # data.fillna(data.mean(), inplace=True)
    # print ('data shapes: ' , data.shape, ' , ', labels.shape, ' , ', foldsdf.shape )




    if ESTIMATORS is None:
        ESTIMATORS = [
                mean_est(),
                RidgeCV(alphas=alphas),
                GradientBoostingRegressor(n_estimators= 200, random_state=1, subsample=0.7, max_depth=6, max_features=0.8), #, min_impurity_decrease=0.05),
                GradientBoostingRegressor(n_estimators= 200, random_state=1, subsample=0.7, max_depth=6, max_features=0.8, min_impurity_decrease=0.1),
                # GradientBoostingRegressor(n_estimators= 250, random_state=2, subsample= subsample, max_depth=max_depth, max_features= max_features),# min_impurity_decrease=0.02),
                # GradientBoostingRegressor(n_estimators= 200, random_state=1, subsample=0.7, max_depth=10, max_features=0.65, min_impurity_decrease=0.05),
                # GradientBoostingRegressor(n_estimators= 250, random_state=2, subsample= subsample, max_depth=max_depth, max_features= max_features, min_impurity_decrease=0.02),
                # RidgeCV(alphas=alphas)
                # BaggingRegressor(n_estimators=20, max_samples=0.9, max_features=0.9, random_state=7),
                # KNeighborsRegressor(n_neighbors=5)
        ]

    ESTIMATORS_NAME = [ 'mean' , 'ridgecv', 'gbr_500' , 'gbr500_i', 'gbr_200_mid', 'gbr_250_mid' ,'ridgecv_res']
    YpredsAll = None
    YpredsAll_controls = None
    # YpredsAllTrain = None
    index = []
    for i in range(folds):

        # prepare train and test data
        test_ids = foldsdf[foldsdf['fold'] == i].index.tolist()
        index = index + test_ids

        print ('residuals : ' , residuals)
        if residuals:
            [ X, Xtrain, Xtest, ytrain , ytest] = split_train_test(controls, labels, foldsdf, i)
        else:
            [ X, Xtrain, Xtest, ytrain , ytest] = split_train_test(data, labels, foldsdf, i, dim_sizes= dim_sizes)

        # return None





        # print ('train & test: ' , Xtrain.shape, ' , ', ytrain.shape , ' , ', Xtest.shape , ' , ', ytest.shape, ' , ', X.shape)

        # [Xtrain, fSelector] = dimension_reduction(Xtrain, ytrain)
        # Xtest = fSelector.transform(Xtest)

        Ypreds = None
        Ypreds_controls = None
        for j in range(len(ESTIMATORS)):
            # print ('j: ' , j, '  , len(ESTIMATORS): ', len(ESTIMATORS))

            # if j==len(ESTIMATORS)-1 and residuals:
            #     [ X, Xtrain, Xtest, ytrain , ytest] = split_train_test(controls, labels, foldsdf, i)

            estimator = ESTIMATORS[j]
            estimator.fit(Xtrain, ytrain)
            ypred = estimator.predict(Xtest)
            ypred = np.reshape(ypred ,newshape =(ypred.shape[0],1))

            if residuals:
                store = True
            else:
                store = False




            if residuals:
                Ypreds_controls = stack_folds_preds(ypred, Ypreds_controls, 'horizontal')
                #####
                ypred_train = estimator.predict(Xtrain)
                ypred_train = np.reshape(ypred_train ,newshape =(ypred_train.shape[0],1))
                evaluate(ytrain, ypred_train, pre='train__', store=store)
                ####

                print ( ytest[:10])
                evaluate(ytest, ypred, pre=pre+'_'+str(i)+'_'+ESTIMATORS_NAME[j]+'_controls_', store=store)
                print ('<<<<<<<<<< residualized control >>>>>>  j : ' , j)
                ypredTrain = estimator.predict(X)
                ypredTrain = np.reshape(ypredTrain ,newshape =(ypredTrain.shape[0],1))
                print ( labels[:10])
                print ( ypredTrain[:10])
                res_labels = np.subtract(labels, ypredTrain)
                print ( res_labels[:10])
                [ X1, Xtrain1, Xtest1, ytrain1 , ytest1] = split_train_test(data, res_labels, foldsdf, i, dim_sizes= dim_sizes)
                print ('...............')
                print ( ytrain1[:10])
                print (',,,,,,,,,,,,,,,,')
                print ( ytest1[:10])
                estimator = RidgeCV(alphas=alphas)
                estimator.fit(Xtrain1, ytrain1)
                ypred1 = estimator.predict(Xtest1)
                print(ypred1[:10])
                ypred1 = np.reshape(ypred1 ,newshape =(ypred1.shape[0],1))

                evaluate(ytest1, ypred1, pre=pre+'_'+str(i)+'_'+ESTIMATORS_NAME[j]+'_residuals_', store=store)

                print ( ypred[:10])
                ypred = np.add(ypred, ypred1)

                # print ( ytest[:10])

            Ypreds = stack_folds_preds(ypred, Ypreds, 'horizontal')
            try:

                evaluate(ytest, ypred, pre=pre+'_'+str(i)+'_'+ESTIMATORS_NAME[j]+'_', store=store)
            except:
                print 'try...except'
                print (ypred.shape)
                print (ytest.shape)
                print (np.isnan(ypred).any())
                print (np.isnan(ytest).any())
                print (labels.isnull().values.any())
                print (labels['fold_'+str(i)].isnull().values.any())

            # if j==1 & residuals:
            #     ypredTrain = estimator.predict(X)
            #     ypredTrain = np.reshape(ypredTrain ,newshape =(ypredTrain.shape[0],1))
            #     YpredsAllTrain = stack_folds_preds(ypredTrain, YpredsAllTrain, 'horizontal')

        Ypreds = stack_folds_preds(ytest, Ypreds, 'horizontal')
        YpredsAll = stack_folds_preds(Ypreds, YpredsAll, 'vertical')
        print ('ypredsAll.shape: ' , YpredsAll.shape)
        if residuals:
            Ypreds_controls = stack_folds_preds(ytest, Ypreds_controls, 'horizontal')
            YpredsAll_controls = stack_folds_preds(Ypreds_controls, YpredsAll_controls, 'vertical')
            print ('ypredsAll.shape: ' , YpredsAll_controls.shape)




    for j in range(YpredsAll.shape[1]-1):
        if residuals:
            evaluate(YpredsAll_controls[:,YpredsAll_controls.shape[1]-1].transpose(), YpredsAll_controls[:,j].transpose(), pre=pre+'_'+ESTIMATORS_NAME[j]+'_controls_')
        evaluate(YpredsAll[:,YpredsAll.shape[1]-1].transpose(), YpredsAll[:,j].transpose(), pre=pre+'_'+ESTIMATORS_NAME[j]+'_')


    result = pd.DataFrame(index=index, data= YpredsAll[:,2], columns=[col_name])

    # return YpredsAllTrain if residuals else result
    return result

def residualized_control(data, controls, labels, foldsdf, folds, pre, scaler=None, n_estimators = 300, subsample=0.75, max_depth=8, max_features = 0.75, residuals = False, col_name='y', ESTIMATORS = None):
    print ('cv...')
    # data.fillna(data.mean(), inplace=True)
    # print ('data shapes: ' , data.shape, ' , ', labels.shape, ' , ', foldsdf.shape )

    if ESTIMATORS is None:
        ESTIMATORS = [
                mean_est(),
                RidgeCV(alphas=alphas),
                # GradientBoostingRegressor(n_estimators= 200, loss='lad', random_state=1, subsample=0.75, max_depth=5, max_features=0.75), #, min_impurity_decrease=0.05),
                # GradientBoostingRegressor(n_estimators= n_estimators, loss='ls', random_state=2, subsample= subsample, max_depth=max_depth, max_features= max_features, min_impurity_decrease=0.02),
                # BaggingRegressor(n_estimators=20, max_samples=0.9, max_features=0.9, random_state=7),
                # KNeighborsRegressor(n_neighbors=5)
        ]

    ESTIMATORS_NAME = [ 'mean' , 'ridgecv', 'gbr_ls' , 'knn' ]
    YpredsAll = None
    YpredsAllTrain = None
    index = []
    for i in range(folds):

        # prepare train and test data
        test_ids = foldsdf[foldsdf['fold'] == i].index.tolist()
        index = index + test_ids


        if residuals:
            [ X, Xtrain, Xtest, ytrain , ytest] = split_train_test(controls, labels, foldsdf, i, dim_reduction=True)
        else:
            [ X, Xtrain, Xtest, ytrain , ytest] = split_train_test(data, labels, foldsdf, i, dim_reduction=True)

        print ('train & test: ' , Xtrain.shape, ' , ', ytrain.shape , ' , ', Xtest.shape , ' , ', ytest.shape, ' , ', X.shape)

        # [Xtrain, fSelector] = dimension_reduction(Xtrain, ytrain)
        # Xtest = fSelector.transform(Xtest)

        Ypreds = None
        for j in range(len(ESTIMATORS)):
            estimator = ESTIMATORS[j]
            estimator.fit(Xtrain, ytrain)
            ypred = estimator.predict(Xtest)
            ypred = np.reshape(ypred ,newshape =(ypred.shape[0],1))


            if j>=1 & residuals:
                ypredTrain = estimator.predict(X)
                ypredTrain = np.reshape(ypredTrain ,newshape =(ypredTrain.shape[0],1))
                labels = np.subtract(labels, ypredTrain)
                [ X1, Xtrain1, Xtest1, ytrain1 , ytest1] = split_train_test(data, labels, foldsdf, i, dim_reduction=True)

                estimator = ESTIMATORS[j]
                estimator.fit(Xtrain1, ytrain1)
                ypred1 = estimator.predict(Xtest1)
                ypred1 = np.reshape(ypred1 ,newshape =(ypred1.shape[0],1))

                ypred = np.add(ypred, ypred1)



            Ypreds = stack_folds_preds(ypred, Ypreds, 'horizontal')
            try:
                evaluate(ytest, ypred, pre=pre+'_'+str(i)+'_'+ESTIMATORS_NAME[j]+'_', store=False)
            except:
                print 'try...except'
                print (ypred.shape)
                print (ytest.shape)
                print (np.isnan(ypred).any())
                print (np.isnan(ytest).any())
                print (labels.isnull().values.any())
                print (labels['fold_'+str(i)].isnull().values.any())


            # if j==1 & residuals:
            #     ypredTrain = estimator.predict(X)
            #     ypredTrain = np.reshape(ypredTrain ,newshape =(ypredTrain.shape[0],1))
            #     YpredsAllTrain = stack_folds_preds(ypredTrain, YpredsAllTrain, 'horizontal')

        Ypreds = stack_folds_preds(ytest, Ypreds, 'horizontal')
        # YpredsAll = stack_folds_preds(Ypreds, YpredsAll, 'vertical')
        print ('ypredsAll.shape: ' , YpredsAll.shape)



    for j in range(YpredsAll.shape[1]-1):
        evaluate(YpredsAll[:,YpredsAll.shape[1]-1].transpose(), YpredsAll[:,j].transpose(), pre=pre+'_'+ESTIMATORS_NAME[j]+'_')

    result = pd.DataFrame(index=index, data= YpredsAll[:,1], columns=[col_name])

    # return YpredsAllTrain if residuals else result
    return result

def only_controls(demog_df, labels_df):
    folds = 10
    demog_df.fillna(demog_df.mean(),inplace=True)
    [demog_df , demog_scaler] = transform(demog_df, type='minmax')#, range=(1,2))
    foldsdf = k_fold(demog_df, index='cnty', folds=folds)
    ESTIMATORS = [
                    mean_est(),
                    RidgeCV(alphas=alphas),
                    GradientBoostingRegressor(n_estimators= 250, loss='ls', random_state=1, subsample=0.75, max_depth=8), #, min_impurity_decrease=0.05),
                ]
    for col in labels_df.columns:
        personality = labels_df[[col]]
        personality.dropna(axis=0, how='any', inplace=True)
        [personality, demog_df, foldsdf] = match_ids([personality, demog_df, foldsdf])
        inferred =  cv(data=[demog_df], controls = demog_df, labels=personality, foldsdf = foldsdf, folds=folds, pre='...controls...'+col+'...', col_name=col, residuals= False , ESTIMATORS=ESTIMATORS)

    exit()


def main():
    print('myMain...')
    ngrams_df, nbools_df, topic_df, demog_df, labels_df = load_data()

    # only_controls(demog_df, labels_df)

    if topic_df is not None:
        print ('topic_df.shape: ', topic_df.shape)
    print ('ses_demog_df.shape: ', demog_df.shape)
    if ngrams_df is not None:
        print ('language_df.shape: ', ngrams_df.shape)
    print ('labels_df.shape: ', labels_df.shape)

    # control_df.to_csv('csv/controls.csv')
    with open("corr_ses_demog_with_disease.txt", "w") as myfile:
        for col in demog_df.columns:

            print ('col:' )
            myfile.write('\ncol: ' + col+ '\n')
            myfile.write( str(labels_df.corrwith(demog_df[col])))
            print (labels_df.corrwith(demog_df[col]))
    #
    # return

    # print ('demog_df.shape after set_index: ', demog_df.shape)

    # language_df = msg_to_user_langauge(language_df)
    if ngrams_df is not None:
        # language_df = run_tfidf_dataframe(language_df, col_name='message')
        print ('language_df(tfidf).shape: ', ngrams_df.shape)
    # language_df.to_csv('language_data.csv')
    # language_df = min_max_transformation(language_df)
    # language_df.to_csv('transformed_data.csv')
    # multiply(demog_df, language_df, output_filename = 'multiplied_transformed_data.csv')

    # res_control(topic_df, language_df, demog_df, personality_df)

    cross_validation(topic_df=topic_df, ngrams_df=ngrams_df, nbools_df=nbools_df, demog_df=demog_df, personality_df=labels_df, folds=5)



def min_max_transformation(data, index_name=''):
    print ('min_max_transformation...')
    if (len(index_name) > 0):
        data.set_index(index_name, inplace=True)
    data = (data - data.min())/(data.max()- data.min())
    print (data.shape)
    data.dropna(axis=1, how='any', inplace=True)
    print (data.shape)
    if (len(index_name) > 0):
        data.reset_index(inplace=True)
    return data
    # scaler = MinMaxScaler()
    # data = scaler.transform(data)
    # return data

if __name__ == '__main__':
    main()