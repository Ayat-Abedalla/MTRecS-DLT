#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 05:19:06 2019

@author: ayat
"""
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

# Set train file names
train_queries_file = "./dataset/train_queries.csv"
train_plans_file = "./dataset/train_plans.csv"
train_click_file= "./dataset/train_clicks.csv"
profiles_file = "./dataset/profiles.csv"

# Set test file names
test_queries_file = "./dataset/test_queries.csv"
test_plans_file = "./dataset/test_plans.csv"
                
def load_prepare_data():
    
    # Load training data
    train_queries = pd.read_csv(train_queries_file)
    train_plans = pd.read_csv(train_plans_file)
    train_click = pd.read_csv(train_click_file)
    
    # Load testing data
    test_queries = pd.read_csv(test_queries_file)
    test_plans = pd.read_csv(test_plans_file)
    
    # Prepare train data
    train_data = train_queries.merge(train_click, on='sid', how='left')
    train_data = train_data.merge(train_plans, on='sid', how='left')
    
    test_data = test_queries.merge(test_plans, on='sid', how='left')
    
    return train_data, test_data


def preprocess_features(train_data, test_data):
    
    train_data = train_data.drop(['click_time'], axis=1)
    train_data['click_mode'] = train_data['click_mode'].fillna(0)
    
    test_data['click_mode'] = -1
    
    # concat train and test sets 
    all_data = pd.concat([train_data, test_data], axis=0, sort=True)
    all_data = all_data.drop(['plan_time'], axis=1)
    all_data = all_data.reset_index(drop=True)
    
    # Prepare OD features by spliting coordinates for each of them
    all_data['o_first'] = all_data['o'].apply(lambda od: float(od.split(',')[0]))
    all_data['o_second'] = all_data['o'].apply(lambda od: float(od.split(',')[1]))
    all_data['d_first'] = all_data['d'].apply(lambda od: float(od.split(',')[0]))
    all_data['d_second'] = all_data['d'].apply(lambda od: float(od.split(',')[1]))
    all_data = all_data.drop(['o', 'd'], axis=1)

    all_data['req_time'] = pd.to_datetime(all_data['req_time'])
    all_data['reqweekday'] = all_data['req_time'].dt.dayofweek
    all_data['reqhour'] = all_data['req_time'].dt.hour
    all_data = all_data.drop(['req_time'], axis=1)
    
    return all_data
    
def generate_plan_features(all_data):
    
    n = all_data.shape[0]
    
    mode_list_feasible = np.zeros((n, 12))
    
    max_distance, min_distance, mean_distance, std_distance = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_price, min_price, mean_price, std_price = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    max_eta, min_eta, mean_eta, std_eta = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))

    mode_min_distance, mode_max_distance, mode_min_price, mode_max_price, mode_min_eta, mode_max_eta, first_mode = np.zeros(
        (n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    
    
    for i, plan in tqdm(enumerate(all_data['plans'].values)):
        
        try:
            plan_list = json.loads(plan)
        except:
            plan_list = []
            
        if len(plan_list) == 0:
            mode_list_feasible[i, 0] = 1
            first_mode[i] = 0

            max_distance[i] = -1
            min_distance[i] = -1
            mean_distance[i] = -1
            std_distance[i] = -1

            max_price[i] = -1
            min_price[i] = -1
            mean_price[i] = -1
            std_price[i] = -1

            max_eta[i] = -1
            min_eta[i] = -1
            mean_eta[i] = -1
            std_eta[i] = -1

            mode_min_distance[i] = -1
            mode_max_distance[i] = -1
            mode_min_price[i] = -1
            mode_max_price[i] = -1
            mode_min_eta[i] = -1
            mode_max_eta[i] = -1
          
        else:
            distance_list = []
            price_list = []
            eta_list = []
            mode_list = []
            
            for tmp_dit in plan_list:
                distance_list.append(int(tmp_dit['distance']))
                if tmp_dit['price'] == '':
                    price_list.append(0)
                else:
                    price_list.append(int(tmp_dit['price']))
                eta_list.append(int(tmp_dit['eta']))
                mode_list.append(int(tmp_dit['transport_mode']))
            
            distance_list = np.array(distance_list)
            price_list = np.array(price_list)
            eta_list = np.array(eta_list)
            mode_list = np.array(mode_list, dtype='int')
            
            mode_list_feasible[i, mode_list] = 1
            
            distance_sort_idx = np.argsort(distance_list)
            price_sort_idx = np.argsort(price_list)
            eta_sort_idx = np.argsort(eta_list)

            max_distance[i] = distance_list[distance_sort_idx[-1]]
            min_distance[i] = distance_list[distance_sort_idx[0]]
            mean_distance[i] = np.mean(distance_list)
            std_distance[i] = np.std(distance_list)

            max_price[i] = price_list[price_sort_idx[-1]]
            min_price[i] = price_list[price_sort_idx[0]]
            mean_price[i] = np.mean(price_list)
            std_price[i] = np.std(price_list)

            max_eta[i] = eta_list[eta_sort_idx[-1]]
            min_eta[i] = eta_list[eta_sort_idx[0]]
            mean_eta[i] = np.mean(eta_list)
            std_eta[i] = np.std(eta_list)

            first_mode[i] = mode_list[0]
            mode_max_distance[i] = mode_list[distance_sort_idx[-1]]
            mode_min_distance[i] = mode_list[distance_sort_idx[0]]

            mode_max_price[i] = mode_list[price_sort_idx[-1]]
            mode_min_price[i] = mode_list[price_sort_idx[0]]

            mode_max_eta[i] = mode_list[eta_sort_idx[-1]]
            mode_min_eta[i] = mode_list[eta_sort_idx[0]]

    feature_data = pd.DataFrame(mode_list_feasible)
    feature_data.columns = ['mode_feasible_{}'.format(i) for i in range(12)]
    feature_data['max_distance'] = max_distance
    feature_data['min_distance'] = min_distance
    feature_data['mean_distance'] = mean_distance
    feature_data['std_distance'] = std_distance

    feature_data['max_price'] = max_price
    feature_data['min_price'] = min_price
    feature_data['mean_price'] = mean_price
    feature_data['std_price'] = std_price

    feature_data['max_eta'] = max_eta
    feature_data['min_eta'] = min_eta
    feature_data['mean_eta'] = mean_eta
    feature_data['std_eta'] = std_eta
    
    feature_data['mode_max_distance'] = mode_max_distance
    feature_data['mode_min_distance'] = mode_min_distance
    feature_data['mode_max_price'] = mode_max_price
    feature_data['mode_min_price'] = mode_min_price
    feature_data['mode_max_eta'] = mode_max_eta
    feature_data['mode_min_eta'] = mode_min_eta
    feature_data['first_mode'] = first_mode
    
    all_data = pd.concat([all_data, feature_data], axis=1)
    all_data = all_data.drop(['plans'], axis=1)
    return all_data

def read_profile_data():
    profile_data = pd.read_csv(profiles_file)
    profile_na = np.zeros(67)
    profile_na[0] = -1
    profile_na = pd.DataFrame(profile_na.reshape(1, -1))
    profile_na.columns = profile_data.columns
    profile_data = profile_data.append(profile_na)
    return profile_data

def generate_profile_features(data):
    profile_data = read_profile_data()
    x = profile_data.drop(['pid'], axis=1).values
    svd = TruncatedSVD(n_components=20, n_iter=20, random_state=42)
    svd_x = svd.fit_transform(x)
    svd_feas = pd.DataFrame(svd_x)
    svd_feas.columns = ['svd_attribute_{}'.format(i) for i in range(20)]
    svd_feas['pid'] = profile_data['pid'].values
    data['pid'] = data['pid'].fillna(-1)
    data = data.merge(svd_feas, on='pid', how='left')
    return data

def split_train_test(data):
    train_data = data[data['click_mode'] != -1]
    test_data = data[data['click_mode'] == -1]
    submit = test_data[['sid']].copy()
    train_data = train_data.drop(['sid', 'pid'], axis=1)
    test_data = test_data.drop(['sid', 'pid'], axis=1)
    test_data = test_data.drop(['click_mode'], axis=1)
    train_y = train_data['click_mode'].values
    train_x = train_data.drop(['click_mode'], axis=1)
    return train_x, train_y, test_data, submit

def save_data(trainX, y_train, testX, y_test):
    trainX.to_csv('preprocess_data/train.csv',index = False)
    testX.to_csv('preprocess_data/test.csv',index = False)
    y_train = pd.DataFrame({'click_mode': y_train})
    y_train.to_csv('preprocess_data/train_label.csv',index = False)
    y_test.to_csv('preprocess_data/test_label.csv',index = False)

def load_data():
    trainX = pd.read_csv('preprocess_data/train.csv')
    testX = pd.read_csv('preprocess_data/test.csv')
    y_train = pd.read_csv('preprocess_data/train_label.csv')
    y_test = pd.read_csv('preprocess_data/test_label.csv')
    return trainX, y_train, testX, y_test

def build_norm_context(trainX, testX):
    trainX = np.array(trainX)
    context_input = trainX[:,:37]
    user_input = trainX[:,37:]
    
    testX = np.array(testX)
    context_input_test = testX[:,:37]
    user_input_test = testX[:,37:]
    
    scaler = MinMaxScaler()
    scaler.fit(context_input)
    
    # apply transform
    normalized_train = scaler.transform(context_input)
    normalized_test = scaler.transform(context_input_test)
    
    normalized_train= pd.DataFrame(normalized_train)
    user_input= pd.DataFrame(user_input)
    merged_train = pd.concat([normalized_train, user_input], axis=1)

    normalized_test= pd.DataFrame(normalized_test)
    user_input_test= pd.DataFrame(user_input_test)
    merged_test = pd.concat([normalized_test, user_input_test], axis=1)
    return merged_train, merged_test

def get_prepare_data(train_data, test_data):
    all_data = preprocess_features(train_data, test_data)
    all_data = generate_plan_features(all_data)
    all_data = generate_profile_features(all_data)
    train_x, train_y, test_x, submit = split_train_test(all_data)
    return train_x, train_y, test_x, submit
    
if __name__ == '__main__':
    pass
