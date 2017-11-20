

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


user= ''
password = ''
database = 'fb22'
host = ''
msg_table = 'messagesEn'
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


def load_language(cursor):
    print('load_language...')
    sql = "select user_id , message from {0}".format(msg_table)
    query = cursor.execute(sql)
    result =  query.fetchall()
    language_df = pd.DataFrame(data = result, columns = ['user_id' , 'message'])
    return language_df


def load_controls(cursor):
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
        language_df = load_language(cursor)
        control_df = load_controls(cursor)

    return language_df, control_df

def run_tfidf(train_tweets, test_tweets=[]):
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

    train_tfidf = tfidf_vectorizer.fit_transform(train_tweets)
    if (len(test_tweets) != 0):
        test_tfidf = tfidf_vectorizer.transform(test_tweets)
        return train_tfidf, test_tfidf
    else:
        return train_tfidf

def run_tfidf_dataframe(data, index_name=''):
    print ('run_tfidf_dataframe...')
    tfidf = run_tfidf(data.values.tolist())
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
    language_df, control_df = load_data()
    print ('control_df.shape: ', control_df.shape)
    print ('language_df.shape: ', language_df.shape)
    control_df.set_index('user_id', inplace=True)
    print ('control_df.shape after set_index: ', control_df.shape)

    language_df = msg_to_user_langauge(language_df)

    language_df = run_tfidf_dataframe(language_df)

    language_df = min_max_transformation(language_df)

    multiply(control_df, language_df, output_filename = 'multiplied_data.csv')

def multiply(controls, language, output_filename):
    print ('multiply...')
    all_df = language
    for col in controls.columns:
        languageMultiplyC = language.multiply(controls[col], axis="index")
        languageMultiplyC.columns = [ s+'_'+col for s in language.columns]
        all_df = pd.concat([all_df, languageMultiplyC] , axis=1, join='inner')

    all_df.to_csv(output_filename)
    return all_df


def min_max_transformation(data, index_name=''):
    print ('min_max_transformation...')
    if (len(index_name) > 0):
        data.set_index(index_name, inplace=True)
    data = (data - data.min())/(data.max()- data.min())
    data.dropna(axis=1, how='any', inplace=True)
    if (len(index_name) > 0):
        data.reset_index(inplace=True)
    return data
    # scaler = MinMaxScaler()
    # data = scaler.transform(data)
    # return data

if __name__ == '__main__':
    myMain()