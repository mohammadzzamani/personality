import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.decomposition import PCA

class transformation:

    tr_mean = 0
    tr_var = 1

    def transform(self,data, column):
        self.tr_mean = data[column].mean()
        self.tr_var = data[column].var()

        data[column] = data[column].map(lambda x: (x - self.tr_mean)/ self.tr_var )

    def transform_back(self, values):
        values = (values * self.tr_var ) + self.tr_mean
        return values


def stack_folds_preds( pred_fold, pred_all=None, type='vertical'):
    print ('stack_folds_preds...')
    if pred_all is None:
        pred_all = pred_fold
    else:
        try:
            pred_all = np.vstack((pred_all, pred_fold)) if type=='vertical' else np.hstack((pred_all, pred_fold))
        except:
            print ( pred_all.shape, '  ' , pred_fold.shape, ' type: ' , type )
            raise
    return pred_all


def outlier_detection(data, column = 'logerror' , thresh=1):
    print ('outlier_detection...')
    print (data.shape)
    # print type(data)
    # data = data[np.where(data[:,0]<0.9)]
    data = data[abs(data[column] ) < thresh]
    print (data.shape)
    return data


def into_classes(data):
    # print 'data: ' , data
    if abs(data)<=1:
        return 0
    elif data<-1:
        return -1
    else: return 1
    # mid = data[abs(data.logerror)<=1]
    # low = data[data.logerror< -1]
    # high = data[data.logerror> 1]
    # print low.shape, ' , ', mid.shape, ' , ', high.shape



class mean_est:
    def __init__(self,type='regression'):
        self.type = type
        self.mean = None

    def fit(self, X, y):
        print ('fit: ' , X.shape ,  '  , ' , y.shape)
        self.mean = np.mean(y)
        if self.type is 'classification':
            self.mean = np.sign(self.mean)

    def predict(self, X):
        print ('predict: ' , X.shape )
        return np.array([ self.mean for i in range(X.shape[0])])



# def evaluate(Ytrue, Ypred, type='regression',  pre = 'pre ', mea=None, va=None):
#     if not mea is None:
#         Ytrue = transform_back(Ytrue, mea, va)
#         Ypred = transform_back(Ypred, mea, va)
#
#     mae = mean_absolute_error(Ytrue,Ypred)
#     mse = mean_squared_error(Ytrue,Ypred)
#     with open("res.txt", "a") as myfile:
#
#         if type is 'regression':
#             myfile.write(pre + 'mae: ' + str(mae)+ ' , mse: ' + str(mse) + ' \n' )
#             print ('mae: ' , mae, ' , mse: ', mse)
#         elif type is 'classification2':
#             print ('accuracy: ' , (2-mae)/2)
#     return [mae , mse]

def evaluate(Ytrue, Ypred, type='regression',  pre = 'pre ', store=True, scaler=None):
    print ('evaluate...')
    if not scaler is None:
        Ytrue = scaler.inverse_transform(Ytrue)
        Ypred = scaler.inverse_transform(Ypred)

    # print ('before mae')
    mae = mean_absolute_error(Ytrue,Ypred)
    mse = mean_squared_error(Ytrue,Ypred)
    # print ('mae: '  , mae)
    with open("res_with_corr_1kusers_inferred_topicsAndtfidf.txt", "a") as myfile:
        if type is 'regression':
            # print ('type: ' , type)
            yt = np.reshape(Ytrue, (len(Ytrue) , 1))
            yp = np.reshape(Ypred, (len(Ypred) , 1))
            data = pd.DataFrame(data = np.hstack((yt,yp)))

            if store == True:
                myfile.write(pre + '\t mae: \t' + str(mae)+ '\t mse: \t' + str(mse) + '\t corr: \t' + str(data.corr()[0][1]) + ' \n' )
            print (pre, ' mae: ' , mae, ' , mse: ', mse,   '\t corr: \t' ,  str(data.corr()[0][1]))


        elif type is 'classification2':
            print ('accuracy: ' , (2-mae)/2)
    return [mae , mse]


def cats_to_int_1param(data):
        print ('cats_to_int...')
        cat_columns = data.select_dtypes(['category','object']).columns
        print ('cat_columns: ' , cat_columns)

        for col in cat_columns:
            print ('col:  ' , col)
            data[col] = pd.Categorical(data[col])
            data[col] = data[col].astype('category').cat.codes
        return data

def cats_to_int(data1, data2):
        print ('cats_to_int...')
        cat_columns = data1.select_dtypes(['category','object']).columns
        print ('cat_columns: ' , cat_columns)

        for col in cat_columns:
            if col == 'propertyzoningdesc':
                continue
            print ('col:  ' , col)
            l1 = list(set(data1[col].values))
            l2 = list(set(data2[col].values))
            # print 'l1, l2: ' , l1, ' , ', l2
            l = l1 + l2
            l = [ ll for ll in l if ll==ll]
            # print 'l: ' , l
            l = list(set(l))
            print 'len l: ' , len(l)
            # l = list(l1.update(l2))
            # print ('l: ' , l)
            data1[col] = [ l.index(c) if c in l else c for c in data1[col]  ]
            data2[col] = [ l.index(c) if c in l else c for c in data2[col]  ]

        return data1, data2




def add_date_features(df, drop_transactiondate=True):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = (df["transactiondate"].dt.year - 2016)*12 + df["transactiondate"].dt.month
    # df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = (df["transactiondate"].dt.year - 2016)*4 +df["transactiondate"].dt.quarter
    if drop_transactiondate:
        df.drop(["transactiondate"], inplace=True, axis=1)
    return df

def get_submission_format(data):
    print ('get_submission_format ...')
    data = data[['ParcelId']]
    # print (data)
    # cols = ['ParcelId', '10/1/16', '11/1/16', '12/1/16', '10/1/17', '11/1/17', '12/1/17']
    cols = ['ParcelId', '2016-10-1', '2016-11-1', '2016-12-1', '2017-10-1', '2017-11-1', '2017-12-1']
    # pd.Timestamp('2016-09-30')
    for i in range(1 ,len(cols)):
        c = cols[i]
        data[c] = 0
    data.columns = cols
    print (data.columns)
    print (data.shape)
    submission_df = pd.melt(data, id_vars=["ParcelId"],var_name="transactiondate", value_name="logerror")
    print ('submission_df: ')
    submission_df['transactiondate'] = [pd.Timestamp(str(s)) for s in  submission_df['transactiondate']]
    # print (submission_df)
    print (submission_df.shape)
    return submission_df

def prepare_final_submission(submission_df, Ypred, type=0, output_filename='data/final_submission_outlierDetection.csv'):
    print ('prepare_final_submission ...')
    print ('submission_df.columns: ' , submission_df.columns, ' , ', submission_df.shape)

    print ('Ypred: ', Ypred)
    ##### prepare submission dataframe to look like the actual submission file (using pivot_table)
    submission_df['logerror'] = Ypred

    submission_df = submission_df[['logerror']]

    print ('submission_df.columns: ' , submission_df.columns)

    # if ('Date' in submission_df.columns):
    # if type == 0:
    submission_df.reset_index(inplace=True)
    print (submission_df)


    submission_df = submission_df.pivot_table(values='logerror', index='ParcelId', columns='transactiondate')

    submission_df.reset_index(inplace=True)

    submission_df.columns = ['ParcelId' , '201610' , '201710', '201611', '201711', '201612', '201712']

    submission_df = submission_df[['ParcelId' , '201610' ,  '201611', '201612', '201710','201711', '201712' ]]

    submission_df.set_index('ParcelId', inplace=True)


    # else:
    #     cols = ['201610' , '201611', '201612', '201710', '201711', '201712']
    #     for i in range(len(cols)):
    #         c = cols[i]
    #         submission_df[c] = submission_df['logerror']
    #     submission_df = submission_df[cols]


    print ('final_submission_df.shape: ' , submission_df.shape)
    print ('final_submission_df.columns: ' , submission_df.columns)
    print (submission_df)
    # final_submission_name = 'data/final_submission_outlierDetection.csv'

    submission_df.to_csv(output_filename)



def match_ids(dataList):
    print ('match_ids...')
    users = None
    for data in dataList:
        # users = pd.DataFrame(index=data.index) if users is None else pd.merge(users, data, how='inner', left_index=True, right_index=True)
        users = data.index if users is None else users.intersection(data.index)

    for i in range(len(dataList)):
        # dataList[i] = all_df[[col for col in all_df.columns if col in dataList[i].columns]]
        dataList[i] = dataList[i].loc[users]

    return dataList


def multiply(controls, language, output_filename=None,  all_df = None):
    print ('multiply...')

    if all_df is None:
        all_df = language
    print ('shapes: ', controls.shape, language.shape , all_df.shape)
    for col in controls.columns:
        print ( col ,  '  , ' , controls[col].shape, '  ,  ', language.shape, '  , ' , all_df.shape)
        languageMultiplyC = language.multiply(controls[col], axis="index")
        languageMultiplyC.columns = [ str(s)+'_'+str(col) for s in language.columns]
        all_df = pd.concat([all_df, languageMultiplyC] , axis=1, join='inner')


    if  output_filename is not None:
        all_df.to_csv(output_filename)
    print ('multiplied_df.shape: ' , all_df.shape)
    # print (all_df.iloc[[0,1]])
    return all_df


def split_train_test(groupData, groupLabels, foldsdf, fold, dim_reduction=None):
    print ('split_train_test...')
    test_ids = foldsdf[foldsdf['fold'] == fold].index.tolist()
    train_ids = foldsdf[foldsdf['fold'] != fold].index.tolist()


    ytrain = groupLabels.loc[train_ids].values  if groupLabels.shape[1] <= 1 else groupLabels['fold_'+str(fold)].loc[train_ids].values
    ytrain = np.reshape(ytrain,(ytrain.shape[0], 1))
    ytest = groupLabels.loc[test_ids].values if groupLabels.shape[1] <= 1 else groupLabels['fold_'+str(fold)].loc[test_ids].values
    ytest = np.reshape(ytest,(ytest.shape[0], 1))
    print ('train & test: ' , ytrain.shape , ' , ', ytest.shape)

    groupX = None
    groupXtrain = None
    groupXtest = None

    for i in range(len(groupData)):
        print ' >>>> i: ' , i
        data = groupData[i]
        X = data.values
        Xtrain = data.loc[train_ids].values
        Xtest = data.loc[test_ids].values
        # index = index + test_ids

        print ( data.shape, ' , ', X.shape, ' , ', Xtrain.shape , ' , ', Xtest.shape)

        if dim_reduction is not None:
            print (np.isnan(Xtrain).any())
            print (np.isnan(ytrain).any())
            print (data.isnull().values.any())
            [Xtrain , fSelector] = dimension_reduction(Xtrain, ytrain)
            Xtest = fSelector.transform(Xtest)
            X = fSelector.transform(X)

        stack_folds_preds(Xtrain, groupXtrain, 'horizontal')
        stack_folds_preds(Xtest, groupXtest, 'horizontal')
        stack_folds_preds(X, groupX, 'horizontal')



    print ('train & test: ' , groupX.shape, ' , ', groupXtrain.shape , ' , ', groupXtest.shape , ' , ', ytest.shape, ' , ', ytrain.shape)

    return groupX, groupXtrain, groupXtest, ytrain , ytest


def dimension_reduction(X, y, univariate = True, pca = True):
    print ('dimension_reduction...')

    alpha = 60.0
    n_components = 100
    if univariate & pca:
        featureSelectionString = 'Pipeline([ ("1_univariate_select", SelectFwe(f_regression, alpha='+str(alpha)+')), ("2_rpca", PCA(n_components='+str(n_components)+', random_state=42, whiten=False, iterated_power=3))])'
    elif univariate:
        featureSelectionString = 'Pipeline([ ("1_univariate_select", SelectFwe(f_regression, alpha='+str(alpha)+'))])'
    elif pca:
        featureSelectionString = 'Pipeline([ ("1_rpca", PCA(n_components='+str(n_components)+', random_state=42, whiten=False, iterated_power=3))])'
    fSelector = eval(featureSelectionString)
    newX = fSelector.fit_transform(X, y)

    print ('newX.shape: ' , newX.shape)

    return [newX , fSelector]