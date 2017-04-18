import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import hashlib
import random
from math import exp
import xgboost as xgb
from sklearn.decomposition import PCA
from math import sin, cos, sqrt, atan2, radians


def dist(list_one, list_two):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(list_one['latitude'])
    lon1 = radians(list_one['longitude'])
    lat2 = radians(list_two['latitude'])
    lon2 = radians(list_two['longitude'])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    
    return distance

def preprocess(train_df, test_df):
    """Just a generic preprocessing function, feel free to substitute it with your custom function"""
    # encode target variable
    train_df['interest_level'] = train_df['interest_level'].apply(lambda x: {'high': 2, 'medium': 1, 'low': 0}[x])
   
    index=list(range(train_df.shape[0]))
    random.shuffle(index)
    manager_score = [np.nan]*len(train_df)
    manager_low = [np.nan]*len(train_df)
    manager_medium = [np.nan]*len(train_df)
    manager_high = [np.nan]*len(train_df)
    manager_low_pct = [np.nan]*len(train_df)
    manager_medium_pct = [np.nan]*len(train_df)
    manager_high_pct = [np.nan]*len(train_df)
    for j in range(5):
        print j
        manager_sum = {}
        manager_high_tmp = {}
        manager_medium_tmp = {}
        manager_low_tmp = {}
        manager_count = {}
        high_total = 0
        medium_total = 0
        low_total = 0
        manager_ct = 0 
        sm = 0
        ct = 0        
        test_ind = index[int((j*train_df.shape[0])/5):int(((j+1)*train_df.shape[0])/5)]
        train_ind = list(set(index).difference(test_ind))
        print 'train ind'
        for i in train_ind:
            x = train_df.iloc[i]
            if x['manager_id'] not in manager_sum:
                manager_sum[x['manager_id']] = 0
                manager_count[x['manager_id']] = 0
                manager_ct += 1
            manager_sum[x['manager_id']] += x['interest_level']
            if  x['interest_level'] == 0:
                if x['manager_id'] not in manager_low_tmp:
                    manager_low_tmp[x['manager_id']] = 0
                manager_low_tmp[x['manager_id']] += 1
                low_total += 1.0
            if  x['interest_level'] == 1:
                if x['manager_id'] not in manager_medium_tmp:
                    manager_medium_tmp[x['manager_id']] = 0
                manager_medium_tmp[x['manager_id']] += 1
                medium_total += 1.0
            if  x['interest_level'] == 2:
                if x['manager_id'] not in manager_high_tmp:
                    manager_high_tmp[x['manager_id']] = 0
                manager_high_tmp[x['manager_id']] += 1
                high_total += 1.0
            manager_count[x['manager_id']] += 1.0
            sm += x['interest_level']        
            ct += 1.0
        avg = sm / ct
        print 'test ind'
        print len(test_ind)
        k = 0
        for i in test_ind:
            x = train_df.iloc[i]
            manager_id = x['manager_id']         
            
            manager_score[i] = manager_sum[manager_id] / manager_count[manager_id] if manager_id in manager_count else avg
            manager_low[i] = manager_low_tmp[manager_id]  if manager_id in manager_low_tmp else low_total / manager_ct
            manager_medium[i] = manager_medium_tmp[manager_id] if manager_id in manager_medium_tmp else medium_total / manager_ct
            manager_high[i] = manager_high_tmp[manager_id] if manager_id in manager_high_tmp  else high_total / manager_ct
            manager_low_pct[i] = manager_low_tmp[manager_id] / manager_count[manager_id]  if manager_id in manager_low_tmp else low_total / ct
            manager_medium_pct[i] = manager_medium_tmp[manager_id] / manager_count[manager_id] if manager_id in manager_medium_tmp else medium_total / ct
            manager_high_pct[i] = manager_high_tmp[manager_id] / manager_count[manager_id] if manager_id in manager_high_tmp else high_total / ct
            if k % 100 == 0:
                print k
            k += 1
   
    train_df['manager_score'] = manager_score 
    train_df['manager_low'] = manager_low 
    train_df['manager_medium'] = manager_medium
    train_df['manager_high'] =  manager_high
    train_df['manager_low_pct'] = manager_low_pct
    train_df['manager_medium_pct'] = manager_medium_pct
    train_df['manager_high_pct'] = manager_high_pct

    train_index = train_df.index
    test_index = test_df.index
    
   
    manager_score = []
    manager_low = []
    manager_medium = []
    manager_high = []
    manager_low_pct = []
    manager_medium_pct = []
    manager_high_pct =[]

    manager_sum = {}
    manager_high_tmp = {}
    manager_medium_tmp = {}
    manager_low_tmp = {}
    manager_count = {}
    building_sum = {}
    building_count = {}
    high_total = 0
    medium_total = 0
    low_total = 0
    manager_ct = 0 
    sm = 0
    ct = 0        
    print 'cv statistics computed'
    for j in range(train_df.shape[0]):
        x=train_df.iloc[j]
        if x['manager_id'] not in manager_sum:
            manager_sum[x['manager_id']] = 0
            manager_count[x['manager_id']] = 0
            manager_ct += 1
        manager_sum[x['manager_id']] += x['interest_level']
        if  x['interest_level'] == 0:
            if x['manager_id'] not in manager_low_tmp:
                manager_low_tmp[x['manager_id']] = 0
            manager_low_tmp[x['manager_id']] += 1
            low_total += 1.0
        if  x['interest_level'] == 1:
            if x['manager_id'] not in manager_medium_tmp:
                manager_medium_tmp[x['manager_id']] = 0
            manager_medium_tmp[x['manager_id']] += 1
            medium_total += 1.0
        if  x['interest_level'] == 2:
            if x['manager_id'] not in manager_high_tmp:
                manager_high_tmp[x['manager_id']] = 0
            manager_high_tmp[x['manager_id']] += 1
            high_total += 1.0
        manager_count[x['manager_id']] += 1.0
        if x['building_id'] not in building_count:
            building_count[x['building_id']] = 0
        building_count[x['building_id']] += 1.0
        sm += x['interest_level']        
        ct += 1.0
        avg = sm / ct
    for i in test_df['manager_id'].values:
        manager_id = i
        manager_score.append(manager_sum[manager_id] / manager_count[manager_id] if manager_id in manager_count else avg)
        manager_low.append(manager_low_tmp[manager_id]  if manager_id in manager_low_tmp else low_total / manager_ct)
        manager_medium.append(manager_medium_tmp[manager_id] if manager_id in manager_medium_tmp else medium_total / manager_ct)
        manager_high.append(manager_high_tmp[manager_id] if manager_id  in manager_high_tmp  else high_total / manager_ct)
        manager_low_pct.append(manager_low_tmp[manager_id] / manager_count[manager_id]  if manager_id in manager_low_tmp else low_total / ct)
        manager_medium_pct.append(manager_medium_tmp[manager_id] / manager_count[manager_id] if manager_id in manager_medium_tmp else medium_total / ct)
        manager_high_pct.append(manager_high_tmp[manager_id] / manager_count[manager_id] if manager_id in manager_high_tmp else high_total / ct)   

    test_df['manager_score'] = manager_score 
    test_df['manager_low'] = manager_low 
    test_df['manager_medium'] = manager_medium
    test_df['manager_high'] =  manager_high
    test_df['manager_low_pct'] = manager_low_pct
    test_df['manager_medium_pct'] = manager_medium_pct
    test_df['manager_high_pct'] = manager_high_pct


    data_df = pd.concat((train_df, test_df), axis=0) 
 
    manager_price = {}
    manager_count = {}
    for j in range(data_df.shape[0]):  
        x=data_df.iloc[j]
        if x['manager_id'] not in manager_price:
            manager_price[x['manager_id']] = 0
            manager_count[x['manager_id']] = 0
        manager_price[x['manager_id']] += x['price']
        manager_count[x['manager_id']] += 1
    
    data_df['avg_manager_price'] = data_df['manager_id'].apply(lambda x: manager_price[x] / manager_count[x])
    # add counting features 
    data_df['num_photos'] = data_df['photos'].apply(len)
    data_df['num_features'] = data_df['features'].apply(len)
    data_df['num_description'] = data_df['description'].apply(lambda x: len(x.split(' ')))
    data_df['photo_description_ratio'] =    data_df['num_photos'] * 1.0 / data_df['num_description']
    data_df.drop('photos', axis=1, inplace=True)
    # naive feature engineering
    data_df['room_difference'] = data_df['bedrooms'] - data_df['bathrooms']
    data_df['room_ratio'] = data_df['bedrooms'] * 1.0 / data_df['bathrooms']
    data_df['total_rooms'] = data_df['bedrooms'] + data_df['bathrooms']
    data_df['price_per_room'] = data_df['price'] / (data_df['total_rooms'] + 1)
    # add datetime features
    data_df['created'] = pd.to_datetime(data_df['created'])
    data_df['c_month'] = data_df['created'].dt.month
    data_df['c_day'] = data_df['created'].dt.day
    data_df['c_hour'] = data_df['created'].dt.hour
    data_df['c_dayofyear'] = data_df['created'].dt.dayofyear
    data_df.drop('created', axis=1, inplace=True)  

    # encode categorical features
    for col in ['display_address', 'street_address', 'manager_id', 'building_id']:
        data_df[col] = LabelEncoder().fit_transform(data_df[col])
    data_df.drop('description', axis=1, inplace=True)
    #data_df.drop('display_address', axis=1, inplace=True)
    #data_df.drop('street_address', axis=1, inplace=True)
    #data_df.drop('building_id', axis=1, inplace=True)
    #data_df.drop('manager_id', axis=1, inplace=True)
    # get text features
    data_df['features'] = data_df['features'].apply(lambda x: ' '.join(['_'.join(i.split(' ')) for i in x]))
    textcv = CountVectorizer(stop_words='english', max_features=100)
    text_features = pd.DataFrame(textcv.fit_transform(data_df['features']).toarray(),
                                                               columns=['f_' + format(x, '03d') for x in range(1, 101)], index=data_df.index)
   #pca = PCA(n_components=20)
    #text_features = pd.DataFrame(pca.fit_transform(text_features))
    #print text_features
    data_df = pd.concat(objs=(data_df, text_features), axis=1)
    data_df.drop('features', axis=1, inplace=True)
    feature_cols = [x for x in data_df.columns if x not in {'interest_level'}]
    del train_df, test_df
    return data_df.loc[train_index, feature_cols], data_df.loc[train_index, 'interest_level'],\
        data_df.loc[test_index, feature_cols]

train = pd.read_json(open("train.json", "r"))
test = pd.read_json(open("test.json", "r"))
train_X, train_y, test_df = preprocess(train, test)
train_X.drop('listing_id', axis=1, inplace=True)
param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.02
param['max_depth'] = 6
param['silent'] = 1
param['num_class'] = 3
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 3
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['seed'] = 321
param['nthread'] = 4
param['num_rounds'] = 1137

print 'training'
xgtrain = xgb.DMatrix(train_X, label=train_y)
#xgb.cv(param, xgtrain, 10000, nfold=3, verbose_eval = True, early_stopping_rounds=10)

model = xgb.train(param, xgtrain, 1137, verbose_eval = True)
listing_id = test_df['listing_id'].ravel()
test_df.drop('listing_id', axis=1, inplace=True)
xgtest = xgb.DMatrix(test_df)

preds = model.predict(xgtest)
sub = pd.DataFrame(data = {'listing_id': listing_id})
sub['low'] = preds[:, 0]
sub['medium'] = preds[:, 1]
sub['high'] = preds[:, 2]
sub.to_csv("submission2.csv", index = False, header = True)


# we simply have to run the following code each time we modify the hyperparameters:
X = cross_validate_lgbm()
