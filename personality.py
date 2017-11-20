

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


user= ''
password = ''
database = 'fb22'
host = ''
msg_table = 'messagesEn'
topic_table = 'feat$cat_met_a30_2000_cp_w$'+msg_table+'$user_id$16to16$r5k'
control_table = 'masterstats'
personality_feats = ['big5_ope', 'big5_ext', 'big5_neu', 'big5_agr', 'big5_con']
demog_feats = ['demog_age_fixed', 'demog_gender']
control_feats = personality_feats + demog_feats

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


def load_tweets(cursor):
    print('load_tweets...')
    sql = "select user_id , message from {0}".format(msg_table)
    query = cursor.execute(sql)
    result =  query.fetchall()
    language_df = pd.DataFrame(data = result, columns = ['user_id' , 'message'])
    return language_df


def load_1to3grams(cursor):
    print('load_1to3grams...')
    # sql = "select user_id , message from {0}".format(msg_table)
    # query = cursor.execute(sql)
    # result =  query.fetchall()
    # language_df = pd.DataFrame(data = result, columns = ['user_id' , 'message'])
    # return language_df

def load_topics(cursor, gft = 500):
    # print('load_topics...')
    # sql = "select distinct(group_id) from {0}".format(topic_table)
    # query = cursor.execute(sql)
    # user_ids =  query.fetchall()
    #
    # topic_df = None
    # counter = 0
    # for user_id in user_ids:
    #     user_id = user_id[0]
    #     counter+=1
    #     sql = 'select group_id , feat, value, group_norm from {0} where group_id = \'{1}\' '.format(topic_table, user_id)
    #     query = cursor.execute(sql)
    #     result =  query.fetchall()
    #     result_df = pd.DataFrame(data = result, columns = ['user_id', 'feat', 'value', 'group_norm'])
    #     uwt = result_df.value.sum()
    #     if counter % gft == 1:
    #         print (sql)
    #         print ('uwt: ' , uwt, ' , ', result_df.shape)
    #     if uwt >= gft:
    #         if topic_df is not None:
    #             topic_df = pd.concat([topic_df,result_df])
    #         else:
    #             topic_df = result_df
    #     if counter % gft == 0:
    #         print (counter , '  ' , topic_df.shape)

    sql = "select group_id , feat, group_norm from {0}".format(topic_table)
    query = cursor.execute(sql)
    result =  query.fetchall()
    topic_df = pd.DataFrame(data = result, columns = ['user_id' , 'feat', 'group_norm'])
    print ('topic_df.shape: ' , topic_df.shape)
    topic_df = topic_df.pivot(index='user_id', columns='feat', values='group_norm')
    print ('topic_df.shape after pivot: ' , topic_df.shape)
    return topic_df

def load_controls(cursor, control_feats = control_feats):
    print('load_controls...')
    feats_str  = ','.join(control_feats)
    sql = "select user_id , {0} from {1}".format(feats_str, control_table)
    query = cursor.execute(sql)
    result =  query.fetchall()
    control_df = pd.DataFrame(data = result, columns = ['user_id'] + control_feats)
    return control_df


def msg_to_user_langauge(language_df):
    print ('msg_to_user_langauge...')
    print ('language_df.shape: ' , language_df.shape)
    language_df = language_df.groupby('user_id').agg({'message':lambda x:' '.join(x)})
    print ('language_df.shape after group by: ' , language_df.shape)
    print ('language_df.columns: ' , language_df.columns)
    return language_df


def load_data():
    print('load_data...')
    try:
        cursor = connectToDB()
    except:
        print("error while connecting to database:", sys.exc_info()[0])
        raise
    if(cursor is not None):
        topic_df = load_topics(cursor)
        # language_df = load_tweets(cursor)
        control_df = load_controls(cursor, control_feats)
        demog_df = load_controls(cursor, demog_feats)

    return topic_df, control_df, demog_df

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

    with open(pickle_name, 'wb') as fin:
        pickle.dump(tfidf_vectorizer, fin)

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
    print ('tfidf.shape: ', len(tfidf) )
    if (len(index_name) > 0):
        data = pd.DataFrame(data = tfidf, index = data[index_name])
    else:
        data = pd.DataFrame(data = tfidf, index = data.index)
    print ( 'data.shape: ' , data.shape)
    return data


def run_kmeans(train_tfidf, d):
    print ('run_kmeans...')
    km = KMeans(n_clusters=d, random_state=123, max_iter=1000,
                n_init=10)
    km.fit(train_tfidf)

    return km.cluster_centers_


def output_factors(train_uids, train_tfidf, test_uids, test_tfidf, cluster_centers,
                   output_f):
    print ('output_factors...')
    uid_list = [train_uids, test_uids]
    feat_list = [train_tfidf, test_tfidf]

    with open(output_f, "w") as f:
        header = ["UID"] + ["Factor {}".format(i + 1) for i in range(len(cluster_centers))]
        f.write("\t".join(header) + "\n")
        for i in range(len(uid_list)):
            us = uid_list[i]
            dists = euclidean_distances(feat_list[i], cluster_centers)
            for j in range(len(us)):
                dist = list(dists[j])
                row = [us[j]] + [str(1 / (factor + .01)) for factor in dist]
                f.write("\t".join(row) + "\n")


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


def main():
    print ('main...')
    input_f, output_f, d = parse_cmd_input()
    train_tweets, train_uids, test_tweets, test_uids = load_data(input_f)

    print "Running tf-idf"
    train_tfidf, test_tfidf = run_tfidf(train_tweets, test_tweets)

    print "Running k-means"
    cluster_centers = run_kmeans(train_tfidf, d)

    output_factors(train_uids, train_tfidf, test_uids, test_tfidf, cluster_centers, output_f)

def myMain():
    print('myMain...')
    language_df, control_df, demog_df = load_data()
    print ('demog_df.shape: ', demog_df.shape)
    print ('control_df.shape: ', control_df.shape)
    print ('language_df.shape: ', language_df.shape)
    control_df.set_index('user_id', inplace=True)
    demog_df.set_index('user_id', inplace=True)
    print ('demog_df.shape after set_index: ', demog_df.shape)

    # language_df = msg_to_user_langauge(language_df)
    # language_df = run_tfidf_dataframe(language_df, col_name='message')
    # language_df.to_csv('language_data.csv')

    # language_df = min_max_transformation(language_df)

    language_df.to_csv('transformed_data.csv')
    #
    multiply(demog_df, language_df, output_filename = 'multiplied_transformed_data.csv')

def multiply(controls, language, output_filename):
    print ('multiply...')
    print ('language.shape: ', language.shape)
    print ('controls.shape: ' , controls.shape)
    all_df = language
    for col in controls.columns:
        languageMultiplyC = language.multiply(controls[col], axis="index")
        languageMultiplyC.columns = [ s+'_'+col for s in language.columns]
        all_df = pd.concat([all_df, languageMultiplyC] , axis=1, join='inner')

    all_df.to_csv(output_filename)
    print ('multiplied_df.shape: ' , all_df.shape)
    print (all_df.iloc[[0,1]])
    return all_df


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
    myMain()