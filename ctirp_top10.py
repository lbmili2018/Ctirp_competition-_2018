#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:58:22 2018

@author: xiaolian
"""

import pandas as pd
import time

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle
import gc
from collections import Counter
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from pylab import *

#========================================================================
'''
    draw line picture
'''
#=========================================================================

def line_plot(x, y):
    plt.plot(x, y)
    #plt.legend()
    plt.show()
#================================================================================================
'''
    eval_metrics
'''
#=================================================================================================

def pr(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    #print(precision)
    #print(recall)
    score = 0
    precision = list(precision)
    recall = list(recall)
    
    for i, value in enumerate(recall):
        if value < 0.5:
            start = i - 1
            break
    for i, value in enumerate(recall):
        if value <= 0.05:
            end = i 
            break    
    for i in range(start, end):
        if i == start:
            score += (0.5 - recall[i+1])*(precision[i] + precision[i+1])
            continue
        if i == end-1:
            score += (recall[i] - 0.05)*(precision[i] + precision[i+1])
            continue            
            
        score += (recall[i] - recall[i+1])*(precision[i] + precision[i+1])
    return 'pr', score/2, True

#================================================================================
'''
    divide time
'''
#================================================================================
def cal_arrival_time_stamp(day):
    #day = day.split(' ')[0]
    #print(day)
    try:
        day = time.mktime(time.strptime(day, '%Y/%m/%d'))
    except:
        day = time.mktime(time.strptime(day, '%Y-%m-%d'))
    #day = time.localtime(day)
    return day

def cal_month(day):
    #day = day.split(' ')[0]
    #print(day)
    try:
        day = time.mktime(time.strptime(day, '%Y/%m/%d'))
    except:
        day = time.mktime(time.strptime(day, '%Y-%m-%d'))
    day = time.localtime(day)
    return day.tm_mon

def cal_week(day):
    #day = day.split(' ')[0]
    #print(day)
    try:
        day = time.mktime(time.strptime(day, '%Y/%m/%d'))
    except:
        day = time.mktime(time.strptime(day, '%Y-%m-%d'))
    day = time.localtime(day)
    return day.tm_wday

# hour
def cal_orderdate_hour(day):
    #day = day.split(' ')[0]
    #print(day)
    try:
        day = time.mktime(time.strptime(day, '%Y/%m/%d %H:%M'))
    except:
        day = time.mktime(time.strptime(day, '%Y-%m-%d %H:%M:%S'))
    day = time.localtime(day)
    return day.tm_hour

# arrival day
def cal_arrival_day(day):
    #day = day.split(' ')[0]
    #print(day)
    try:
        day = time.mktime(time.strptime(day, '%Y/%m/%d'))
    except:
        day = time.mktime(time.strptime(day, '%Y-%m-%d'))
    day = time.localtime(day)
    return day.tm_mday

# orderdate day
def cal_orderdate_day(day):
    #day = day.split(' ')[0]
    #print(day)
    try:
        day = time.mktime(time.strptime(day, '%Y/%m/%d %H:%M'))
    except:
        day = time.mktime(time.strptime(day, '%Y-%m-%d %H:%M:%S'))
    day = time.localtime(day)
    return day.tm_mday

# orderdate mouth
def cal_orderdate_mouth(day):
    #day = day.split(' ')[0]
    #print(day)
    try:
        day = time.mktime(time.strptime(day, '%Y/%m/%d %H:%M'))
    except:
        day = time.mktime(time.strptime(day, '%Y-%m-%d %H:%M:%S'))
    day = time.localtime(day)
    return day.tm_mon

# orderdate week
def cal_orderdate_week(day):
    #day = day.split(' ')[0]
    #print(day)
    try:
        day = time.mktime(time.strptime(day, '%Y/%m/%d %H:%M'))
    except:
        day = time.mktime(time.strptime(day, '%Y-%m-%d %H:%M:%S'))
    day = time.localtime(day)
    return day.tm_wday

# 
def cal_orderdate_time_stamp(day):
    #day = day.split(' ')[0]
    #print(day)
    try:
        day = time.mktime(time.strptime(day, '%Y/%m/%d %H:%M'))
    except:
        day = time.mktime(time.strptime(day, '%Y-%m-%d %H:%M:%S'))
    #day = time.localtime(day)
    return day
#================================================================================
'''
    get data
'''
#================================================================================

def get_data():
    order_train = pd.read_csv('../data/train/ord_train.csv')
    #order_chaifen = pd.read_csv('./data/train/ord_chaifen.csv')
    #order_bkroomstatus = pd.read_csv('./data/train/ord_bkroomstatus.csv')
    #order_zqroomstatus = pd.read_csv('./data/train/ord_zqroomstatus.csv')
    hotelinfo = pd.read_csv('../data/train/hotelinfo.csv')
    #mroominfp = pd.read_csv('./data/train/mroominfo.csv')
    #mhotelinfo = pd.read_csv('./data/train/mhotelinfo.csv')
    
    testa = pd.read_csv('../data/test/ord_testA.csv', encoding='gb2312')
    testb = pd.read_csv('../data/test/ord_testB.csv', encoding='gb2312')
    
    test = pd.concat([testa, testb])
    
    test['noroom'] = -1
    data = pd.concat([order_train, test])
    
    print('pos / all data rate:', len(data[data.noroom == 1])/len(data[data.noroom != -1]))
    
    data.zone = data.zone.replace(0, -1)
    

    hotelinfo.rename(columns = {'totalrooms':'hotel_totalrooms'}, inplace=True)
    data = pd.merge(data, hotelinfo[['hotel', 'hotel_totalrooms']], on = 'hotel', how = 'left')
    
    data = data.fillna(-1)
    
    print('data load end')
    
    data['time_arrival'] = data.arrival.apply(cal_arrival_time_stamp)
    

    
    data['month_arrival'] = data.arrival.apply(cal_month)
    
    data['week_arrival'] = data.arrival.apply(cal_week)
    
    data['orderdate_hour'] = data.orderdate.apply(cal_orderdate_hour)
    
    data['arrival_day'] = data.arrival.apply(cal_arrival_day)
    
    data['orderdate_day'] = data.orderdate.apply(cal_orderdate_day)
    
    print('half finish!')
    
    data['orderdate_mouth'] = data.orderdate.apply(cal_orderdate_mouth)
    
    data['orderdate_week'] = data.orderdate.apply(cal_orderdate_week)
    
    data['time_orderdate'] = data.orderdate.apply(cal_orderdate_time_stamp)
    
    data['span_orderdate_arrival'] = data.time_arrival - data.time_orderdate
    
    data['eta_arrival'] = pd.to_datetime(data.etd) - pd.to_datetime(data.arrival)
    data.eta_arrival = data.eta_arrival.astype(int) / (3600*24*1000000000)
    
    data[data.noroom != -1].price = data.price / ( data.eta_arrival * data.ordroomnum) 
    
    data = data.sort_values(by = 'time_orderdate')
    
    print('row feature create')
    
    return data
#================================================================================
'''
    count feature
'''
#================================================================================

def count_f(x, count):
    if x in count.keys():
        count[x] += 1
        #print(count[x])
        return count[x] - 1
    else:
        count[x] = 1
        #print(count[x])
        return 0

def feature_count(feature, count):
    new_feature = feature + '_count'
    start = time.time()
    data[new_feature] = data[feature].apply(lambda row: count_f(row, count))
    end = time.time()
    print(new_feature, ' ', end-start)
    
count_features =  [
        #'city',
        #'countryid',
        #'hotel',
        #'zone',
        'room',
        #'room_etd',
        'room_arrival',
        #'room_eta_arrival',
        #'room_week_arrival',
        #'room_orderdate_week',
        #'isholdroom',
        #'arrival',
        #'etd',
        #'masterbasicroomid',
        #'masterhotelid',
        #'supplierid',
        #'isvendor',
        #'hotelbelongto',
        #'isebookinghtl',
        #'supplierchannel',    
    
        # stage1 ================================================================================================
        #'ordadvanceday',
        #'hotelstar',
        #'time_arrival', 
        #'month_arrival', 
        #'week_arrival', 
        #'orderdate_hour',      
        #'arrival_day',
        #'eta_arrival',
        #'orderdate_day', 
        #'orderdate_mouth', 
        #'orderdate_week',
        #'time_orderdate',
        #'span_orderdate_arrival',
        # stage3 ================================================================================================
        'hotel_totalrooms',
]
hotels = [
         'isholdroom',
        'masterbasicroomid',
        'masterhotelid',
        'supplierid',
        'isvendor',
        'hotelbelongto',
        'isebookinghtl',
        'supplierchannel',    
        'city',
        'countryid',
        'hotel',
        'zone',
        'room',
        'hotel_totalrooms',
        'hotelstar',
        ]

users = [
          'ordadvanceday',
       # 'time_arrival', 
        'month_arrival', 
        'week_arrival', 
        #'orderdate_hour',      
        'arrival_day',
        'eta_arrival',
        
        #'orderdate_day', 
        #'orderdate_mouth', 
        #'orderdate_week',

    
        # stage3 ================================================================================================
       
        ]

#======================================================================================
'''
    cvr feature
'''
#=======================================================================================
def count_pos_f(feature, noroom, count):
    #print(noroom)
    if noroom == 0 or noroom == -1:
        # print(count[feature])
        return count[feature]
    elif feature in count.keys():
        count[feature] += 1
        return count[feature] - 1
    else:
        # print(count[feature])
        count[feature] = 1
        return 0
        
def feature_pos_count(feature, count):
    new_feature = feature + '_pos_count'
    start = time.time()
    data[new_feature] = data[[feature, 'noroom']].apply(lambda row: count_pos_f(row[feature], row['noroom'], count), axis = 1)
    #data[feature+'_rate'] = round(data[new_feature]/data[feature+'_count'], 2)
    end = time.time()
    print(new_feature, ' ', end-start)
    
    
#========================================================================================
'''
    divide 
'''
#========================================================================================

def transfer(se, value_counts, split = 10):
    eq = len(value_counts)/split
    print(eq)
    # equal frequency
    def eqval(x):
        #print((int(value_counts[x])//eq + 1))
        return (-(value_counts[x]//eq) + split)
    return se.apply(eqval)

def value_counts_eum(value_count):
    count = 1
    
    for i,v in value_count.items():
        value_count[i] = count
        count += 1
    return value_count

#===========================================================================
'''
    cal_pre_order_room_num
'''
#==========================================================================

#============================================================================    
'''
    new feature create
'''
#==========================================================================
def new_feature_create(data):
    
    data['room_arrival'] = data.room.astype(str) + '_' + data.arrival.astype(str)
    
    '''
    for user in users:
        for hotel in hotels:
            new_feature = user + '_' + hotel
            data[new_feature] = data[user].astype(str) + '_' + data[hotel].astype(str)
    '''
    
    for feature in count_features:
        count = Counter()
        feature_count(feature, count)
    
    #vc = value_counts_eum(data.ordadvanceday.value_counts())
    #data['ordadvanceday_10_split'] = transfer(data['ordadvanceday'], vc, 8)
    
    #print('ordadvanceday_10_split create')
    
    cal = data[(data.orderdate < '2017-09-01 00:00:00')]
    cal_data = data[data.orderdate >= '2017-09-01 00:00:00']
    cal_data_train = cal_data[cal_data.noroom != -1]
    cal_data_test = cal_data[cal_data.noroom == -1]
    
    '''
    arrival_count = {}
    for index, row in cal.iterrows():
        if row['noroom'] == 0:
            time_arrival = row['time_arrival']
            for i in range(int(row['eta_arrival'])):
                if time_arrival not in arrival_count.keys():
                    arrival_count[time_arrival] = 1
                else:
                    arrival_count[time_arrival] += 1
                time_arrival += 86400
                    
    def cal_arrival_count(time_arrival, noroom, eta_arrival):
        t = time_arrival
        if noroom == 0:
            for i in range(int(eta_arrival)):
                if time_arrival not in arrival_count.keys():
                    arrival_count[time_arrival] = 1
                else:
                    arrival_count[time_arrival] += 1
                time_arrival += 86400
        
        arr_count = []
        for i in range(int(eta_arrival)):
            if t in arrival_count.keys():
                arr_count.append(arrival_count[t])
            t += 86400
        
        if len(arr_count) == 0:
            return 0
        else:
            return max(arr_count)
        
    cal_data_train['arrival_count'] = cal_data_train.apply(lambda row: cal_arrival_count(row['time_arrival'] ,row['noroom'], row['eta_arrival']),axis = 1)
    cal_data_test['arrival_count'] = cal_data_test.apply(lambda row: cal_arrival_count(row['time_arrival'] ,row['noroom'], row['eta_arrival']),axis = 1)    
    '''       
        
    
    cal_pre_order_num = {}
    
    # cal_pre_order_failure_num = {}
    for index, row in cal.iterrows():
        if row['noroom'] == 0:
            time_arrival = row['time_arrival']
            for i in range(int(row['eta_arrival'])):
                if row['hotel'] not in cal_pre_order_num.keys():
                    order = {}
                    order[time_arrival] = 1
                    cal_pre_order_num[row['hotel']] = order
                else:
                    if time_arrival not in cal_pre_order_num[row['hotel']].keys():
                        cal_pre_order_num[row['hotel']][time_arrival] = 1
                    else:
                        cal_pre_order_num[row['hotel']][time_arrival] += 1
                        
                time_arrival += 86400     
    
    can_pre_order_num = cal_pre_order_num.copy()
    
    def cal_pre_order_room_num(room, time_arrival, noroom, eta_arrival):
        if room not in cal_pre_order_num.keys():
            '''
            if noroom == 0:
                
                for i in range(int(eta_arrival)):
                    if room not in cal_pre_order_num.keys():
                        
                        order = {}
                        order[time_arrival] = 1
                        cal_pre_order_num[room] = order
                        
                    elif time_arrival not in cal_pre_order_num[room].keys():
                        cal_pre_order_num[room][time_arrival] = 1
                    time_arrival += 86400
            '''
            return -1        
        
        elif time_arrival not in cal_pre_order_num[room].keys():
            '''
            if noroom == 0:
                for i in range(int(eta_arrival)):
                    if time_arrival not in cal_pre_order_num[room].keys():
                        cal_pre_order_num[room][time_arrival] = 1
                    else:
                        cal_pre_order_num[room][time_arrival] += 1
                    time_arrival += 86400
            '''
            return -1
        else:
            '''
            if noroom == 0:
                cal_pre_order_num[room][time_arrival] += 1
            '''
            return cal_pre_order_num[room][time_arrival] + 1    
 
    cal_data_train['pre_room_order_num'] = cal_data_train[['hotel', 'time_arrival', 'noroom', 'eta_arrival']].apply(lambda row:cal_pre_order_room_num(row['hotel'], row['time_arrival'], row['noroom'], row['eta_arrival']) , axis = 1)
    cal_data_test['pre_room_order_num'] = cal_data_test[['hotel', 'time_arrival', 'noroom', 'eta_arrival']].apply(lambda row:cal_pre_order_room_num(row['hotel'], row['time_arrival'], row['noroom'], row['eta_arrival']) , axis = 1)
    print('pre_room_order_num create')

    count = Counter()
    for room_key, room_value in cal_pre_order_num.items():
        h = 0
        for t_key, k_value in room_value.items():
            if k_value > h:
                h = k_value
        #_total.append(h)
        count[room_key] = h    
    
    cal_data_train['can_pre_order_room_num'] = cal_data_train.hotel.apply(lambda x: count[x] if x in count.keys() else -1) 
    
    
    count = Counter()
    for room_key, room_value in can_pre_order_num.items():
        h = 0
        for t_key, k_value in room_value.items():
            if k_value > h:
        #_total.append(h)
 
                h = k_value
        count[room_key] = h
   
    cal_data_test['can_pre_order_room_num'] = cal_data_test.hotel.apply(lambda x: count[x] if x in count.keys() else -1)
    print('can_pre_order_room_num create')
    
    #cal_pre_order_num = {}
    cal_pre_order_failure_num = {}
    for index, row in cal.iterrows():
        if row['noroom'] == 1:
            
            time_arrival = row['time_arrival']
            
            for i in range(int(row['eta_arrival'])):
                if row['room'] not in cal_pre_order_failure_num.keys():
                    order = {}
                    order[time_arrival] = 1
                    cal_pre_order_failure_num[row['room']] = order
                else:
                    if time_arrival not in cal_pre_order_failure_num[row['room']].keys():
                        cal_pre_order_failure_num[row['room']][time_arrival] = 1
                    else:
                        cal_pre_order_failure_num[row['room']][time_arrival] += 1
                        
                time_arrival += 86400    
    
    def cal_pre_order_failure_room_num(room, time_arrival, noroom, eta_arrival):
        if room not in cal_pre_order_failure_num.keys():
            if noroom == 1:
                
                for i in range(int(eta_arrival)):
                    if room not in cal_pre_order_failure_num.keys():
                        
                        order = {}
                        order[time_arrival] = 1
                        cal_pre_order_failure_num[room] = order
                        
                    elif time_arrival not in cal_pre_order_failure_num[room].keys():
                        cal_pre_order_failure_num[room][time_arrival] = 1
                    time_arrival += 86400
            return 0        
        
        elif time_arrival not in cal_pre_order_failure_num[room].keys():
            
            if noroom == 1:
                for i in range(int(eta_arrival)):
                    if time_arrival not in cal_pre_order_failure_num[room].keys():
                        cal_pre_order_failure_num[room][time_arrival] = 1
                    else:
                        cal_pre_order_failure_num[room][time_arrival] += 1
                    time_arrival += 86400
            return 0
        else:
            return cal_pre_order_failure_num[room][time_arrival]    
        
    cal_data_train['pre_room_order_failure_num'] = cal_data_train[['room', 'time_arrival', 'noroom', 'eta_arrival']].apply(lambda row:cal_pre_order_failure_room_num(row['room'], row['time_arrival'], row['noroom'], row['eta_arrival']) , axis = 1)
    cal_data_test['pre_room_order_failure_num'] = cal_data_test[['room', 'time_arrival', 'noroom', 'eta_arrival']].apply(lambda row:cal_pre_order_failure_room_num(row['room'], row['time_arrival'], row['noroom'], row['eta_arrival']) , axis = 1)
    cal_data_train['remain_room_num'] = cal_data_train.can_pre_order_room_num - cal_data_train.pre_room_order_num
    cal_data_test['remain_room_num'] = cal_data_test.can_pre_order_room_num - cal_data_test.pre_room_order_num
    print('pre_room_order_failure_num create')
    print('remain_room_num create')
    '''
    big_than_anvorder_day = {}
    for index, row in cal.iterrows():
        if row['noroom'] == 0:
            if row['room'] not in big_than_anvorder_day.keys():
                big = []
                big.append(1)
                big.append(row['ordadvanceday'])
                big_than_anvorder_day[row['room']] = big
            else:
                big_than_anvorder_day[row['room']][0] += 1
                big_than_anvorder_day[row['room']][1] += row['ordadvanceday']
                
    def ordadvanceday_average(room, ordadvanceday, noroom):
    
        if room not in big_than_anvorder_day.keys():
            if noroom == 0:
                big = []
                big.append(1)
                big.append(ordadvanceday)
                big_than_anvorder_day[room] = big
            return 0
        else:
            #print(big_than_anvorder_day[room])
            l = big_than_anvorder_day[room][1]/(big_than_anvorder_day[room][0])
            if noroom == 0:
                
                big_than_anvorder_day[room][0] += 1
                big_than_anvorder_day[room][1] += ordadvanceday   
            return l
        
    cal_data_train['room_average_day'] = cal_data_train[['room', 'ordadvanceday', 'noroom']].apply(lambda row:ordadvanceday_average(row['room'], row['ordadvanceday'], row['noroom']), axis = 1)
    cal_data_test['room_average_day'] = cal_data_test[['room', 'ordadvanceday', 'noroom']].apply(lambda row:ordadvanceday_average(row['room'], row['ordadvanceday'], row['noroom']), axis = 1)
    
    cal_data_train.room_average_day = cal_data_train.room_average_day.replace(0, 25.20745920745921)
    cal_data_test.room_average_day = cal_data_test.room_average_day.replace(0, 25.20745920745921)
    
    cal_data_train['rate_advday_aver'] = cal_data_train[['ordadvanceday','room_average_day']].apply(lambda row: row['ordadvanceday']/row['room_average_day'], axis = 1)
    cal_data_test['rate_advday_aver'] = cal_data_test[['ordadvanceday','room_average_day']].apply(lambda row: row['ordadvanceday']/row['room_average_day'], axis = 1)
    print('room_aerage_day')
    '''
    '''
    #cal_pre_order_num = {}
    cal_pre_order_failure_num = {}
    for index, row in cal.iterrows():
        if row['noroom'] == 1:
    
            time_arrival = row['time_arrival']
    
            for i in range(int(row['eta_arrival'])):
                if row['hotel'] not in cal_pre_order_failure_num.keys():
                    order = {}
                    order[time_arrival] = 1
                    cal_pre_order_failure_num[row['hotel']] = order
                else:
                    if time_arrival not in cal_pre_order_failure_num[row['hotel']].keys():
                        cal_pre_order_failure_num[row['hotel']][time_arrival] = 1
                    else:
                        cal_pre_order_failure_num[row['hotel']][time_arrival] += 1
    
                time_arrival += 86400    
    
    def cal_pre_order_failure_hotel_num(room, time_arrival, noroom, eta_arrival):
        if room not in cal_pre_order_failure_num.keys():
            if noroom == 1:
    
                for i in range(int(eta_arrival)):
                    if room not in cal_pre_order_failure_num.keys():
    
                        order = {}
                        order[time_arrival] = 1
                        cal_pre_order_failure_num[room] = order
    
                    elif time_arrival not in cal_pre_order_failure_num[room].keys():
                        cal_pre_order_failure_num[room][time_arrival] = 1
                    time_arrival += 86400
            return 0        
    
        elif time_arrival not in cal_pre_order_failure_num[room].keys():
    
            if noroom == 1:
                for i in range(int(eta_arrival)):
                    if time_arrival not in cal_pre_order_failure_num[room].keys():
                        cal_pre_order_failure_num[room][time_arrival] = 1
                    else:
                        cal_pre_order_failure_num[room][time_arrival] += 1
                    time_arrival += 86400
            return 0
        else:
            return cal_pre_order_failure_num[room][time_arrival]    
    
    cal_data_train['pre_hotel_order_failure_num'] = cal_data_train[['hotel', 'time_arrival', 'noroom', 'eta_arrival']].apply(lambda row:cal_pre_order_failure_hotel_num(row['hotel'], row['time_arrival'], row['noroom'], row['eta_arrival']) , axis = 1)
    cal_data_test['pre_hotel_order_failure_num'] = cal_data_test[['hotel', 'time_arrival', 'noroom', 'eta_arrival']].apply(lambda row:cal_pre_order_failure_hotel_num(row['hotel'], row['time_arrival'], row['noroom'], row['eta_arrival']) , axis = 1)
    
    
    #cal_pre_order_num = {}
    cal_pre_order_failure_num = {}
    for index, row in cal.iterrows():
        if row['noroom'] == 1:
    
            time_arrival = row['time_arrival']
    
            for i in range(int(row['eta_arrival'])):
                if row['masterbasicroomid'] not in cal_pre_order_failure_num.keys():
                    order = {}
                    order[time_arrival] = 1
                    cal_pre_order_failure_num[row['masterbasicroomid']] = order
                else:
                    if time_arrival not in cal_pre_order_failure_num[row['masterbasicroomid']].keys():
                        cal_pre_order_failure_num[row['masterbasicroomid']][time_arrival] = 1
                    else:
                        cal_pre_order_failure_num[row['masterbasicroomid']][time_arrival] += 1
    
                time_arrival += 86400    
    
    def cal_pre_order_failure_masterbasicroomid_num(room, time_arrival, noroom, eta_arrival):
        if room not in cal_pre_order_failure_num.keys():
            if noroom == 1:
    
                for i in range(int(eta_arrival)):
                    if room not in cal_pre_order_failure_num.keys():
    
                        order = {}
                        order[time_arrival] = 1
                        cal_pre_order_failure_num[room] = order
    
                    elif time_arrival not in cal_pre_order_failure_num[room].keys():
                        cal_pre_order_failure_num[room][time_arrival] = 1
                    time_arrival += 86400
            return 0        
    
        elif time_arrival not in cal_pre_order_failure_num[room].keys():
    
            if noroom == 1:
                for i in range(int(eta_arrival)):
                    if time_arrival not in cal_pre_order_failure_num[room].keys():
                        cal_pre_order_failure_num[room][time_arrival] = 1
                    else:
                        cal_pre_order_failure_num[room][time_arrival] += 1
                    time_arrival += 86400
            return 0
        else:
            return cal_pre_order_failure_num[room][time_arrival]    
    
    cal_data_train['pre_masterbasicroomid_order_failure_num'] = cal_data_train[['masterbasicroomid', 'time_arrival', 'noroom', 'eta_arrival']].apply(lambda row:cal_pre_order_failure_masterbasicroomid_num(row['masterbasicroomid'], row['time_arrival'], row['noroom'], row['eta_arrival']) , axis = 1)
    cal_data_test['pre_masterbasicroomid_order_failure_num'] = cal_data_test[['masterbasicroomid', 'time_arrival', 'noroom', 'eta_arrival']].apply(lambda row:cal_pre_order_failure_masterbasicroomid_num(row['masterbasicroomid'], row['time_arrival'], row['noroom'], row['eta_arrival']) , axis = 1)

  
    span_order = {}
    # span time
    for index, row in cal.iterrows():
        if row['noroom'] == 0:
            time_arrival = row['time_arrival']
            for i in range(int(row['eta_arrival'])):
                if row['room'] not in span_order.keys():
                    order = {}
                    order[time_arrival] = row['time_orderdate']
                    span_order[row['room']] = order
                else:
                    span_order[row['room']][time_arrival] = row['time_orderdate']
                time_arrival += 86400   
        
    def span(room, time_arrival, eta_arrival, time_orderdate, noroom):
        re = []
        t = time_arrival
        if room in span_order.keys():
            for i in range(int(eta_arrival)):
                if t in span_order[room].keys():
                    re.append(span_order[room][t])
                t += 86400
    #for index, row in cal.iterrows():
        if noroom == 0:
            #time_arrival = row['time_arrival']
            for i in range(int(eta_arrival)):
                if room not in span_order.keys():
                    order = {}
                    order[time_arrival] = time_orderdate
                    span_order[room] = order
                else:
                    span_order[room][time_arrival] = time_orderdate
                time_arrival += 86400                
        if len(re) == 0:
            return 0
        else:
            return time_orderdate - max(re)
    
    cal_data_train['span_orderdate_time'] = cal_data_train.apply(lambda row:span(row['room'], row['time_arrival'], row['eta_arrival'], row['time_orderdate'], row['noroom']) , axis = 1)
    cal_data_test['span_orderdate_time'] = cal_data_test.apply(lambda row:span(row['room'], row['time_arrival'], row['eta_arrival'], row['time_orderdate'], row['noroom']) , axis = 1)
    
    cal_data_train['span_ot'] = cal_data_train.apply(lambda row:row['span_orderdate_time'] if row['pre_room_order_failure_num'] != 0 else 0, axis = 1)
    cal_data_test['span_ot'] = cal_data_test.apply(lambda row: row['span_orderdate_time'] if row['pre_room_order_failure_num'] != 0 else 0, axis = 1)
    '''
    return cal_data_train, cal_data_test

#==========================================================================================
'''
    train features
'''
#===========================================================================================

one_hot_feature = [
                   # stage1 =======================================================================================
                   'orderdate',
                   'city',
                   'hotel',
                   'zone',
                   'room',
                   'isholdroom',
                   'arrival',
                   'etd',
                   'masterbasicroomid',
                   'countryid',
                   'masterhotelid',
                   'supplierid',
                   'isvendor',
                   'hotelbelongto',
                   'isebookinghtl',
                   'supplierchannel',
                   'room_arrival',
                   #==================================
             #'ordadvanceday_room',
                   
]

cate = [
        # stage1 ================================================================================================
        'ordadvanceday',
        'hotelstar',
        'eta_arrival',
        
        'week_arrival',
        
       'orderdate_week',
       
        'time_orderdate',
        
        'span_orderdate_arrival',
        
        'hotel_totalrooms',
        
        'room_count',
        'pre_room_order_failure_num',
        
        #'pre_room_order_num',
        #'can_pre_order_room_num',
        
        ##'time_arrival', 
        'room_arrival_count',        
        ##'hotel_count',
       # #'pre_hotel_order_failure_num',
        #'pre_masterbasicroomid_order_failure_num',
        #'room_cvr',
        'arrival_day',
        'orderdate_day',
        #'span_ot',
        #'arrival_count',
        ##'orderid',
        ##'orderdate_hour',
        #====================================
  #       'city_count',
 #'countryid_count',
 #'hotel_count',
 #'zone_count',
 #'room_count',
 #'room_arrival_count',
 #'masterbasicroomid_count',
 #'masterhotelid_count',
 ##'supplierid_count',
 #'isvendor_count',
 #'hotelbelongto_count',
 #'isebookinghtl_count',
 #'supplierchannel_count',
 #'ordadvanceday_count',
 #'hotelstar_count',
 #'eta_arrival_count',
 #'hotel_totalrooms_count',
        
]

start = time.time()

#=====================================================================================================================
'''
    main
'''
#======================================================================================================================
'''
data = get_data()

print('get data time : ', time.time() - start)
start = time.time()

train, test = new_feature_create(data)
data = pd.concat([train, test])

print('new_feature_create time : ', time.time() - start)
start = time.time()

print(len(data))
print(one_hot_feature)

data.to_csv('data.csv', index = False)

data = data.fillna(0)
'''
for i in range(2018, 2019):
    data = pd.read_csv('data.csv')
    #cal_data_train['span_ot'] = cal_data_train.apply(lambda row:row['span_orderdate_time'] if row['pre_room_order_failure_num'] != 0 else 0, axis = 1)
    
    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])
    
    #data['pre_can_room_rate'] = data.pre_room_order_num / data.can_pre_order_room_num
    #data = data.fillna(0)
    
    # vc = value_counts_eum(data.ordadvanceday.value_counts())
    # data['ordadvanceday_10_split'] = transfer(data['ordadvanceday'], vc, 8)
    
    testa = data[data.noroom == -1][['orderid','room','arrival', 'noroom']]
    # train data
    train=data[(data.noroom !=-1) & (data.orderdate_day <= 14)]
    print(len(train))
    #for i in ['room','hotel','masterbasicroomid','masterhotelid']:
    #    train[i+'_rate'] = train[[i+'_rate', i+'_count']].apply(lambda row:row[i+'_rate']/5.27 if row[i+'_count'] < 6 else row[i+'_rate'], axis = 1)
    # shu
    valid = data[(data.noroom != -1) & (data.orderdate_day == 14)]
    print(len(valid))
    #train = shuffle(train)
    
    train_y=train.pop('noroom')
    valid_y = valid.pop('noroom')
    
    test=data[data.noroom == -1]
    test=test.drop('noroom',axis=1)
    enc = OneHotEncoder()
    
    train_x = train[cate]
    valid_x = valid[cate]
    test_x = test[cate]
    
    print(' success!')
    print('divide data : ', time.time() - start)
    start = time.time()
    
    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        train_a = enc.transform(train[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))
        valid_a = enc.transform(valid[feature].values.reshape(-1, 1))
        train_x= sparse.hstack((train_x, train_a))
        valid_x = sparse.hstack((valid_x, valid_a))
        test_x = sparse.hstack((test_x, test_a))
            
        del train_a
        del test_a
        del valid_a
        gc.collect()
            
        print(feature+' finish')
    print('one-hot prepared !')
      
    clf = lgb.LGBMClassifier(
            boosting_type='gbdt', subsample=0.70, colsample_bytree=0.70, 
            max_depth=-1, n_estimators=1400, objective='binary',# min_child_weight = 30, 
            subsample_freq=1, num_leaves=31, reg_alpha=0,reg_lambda = 1,
             random_state=2018, n_jobs=-1, learning_rate=0.05
        )
    
    #X_train, X_test, y_train, y_test = train_test_split(train_x,train_y,test_size=0.3, random_state=2018)
    #clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=pr,early_stopping_rounds=100)
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric=pr)
    
    print('train data time : ', time.time() - start)
    start = time.time()
    
    # submission file create
            
    testa['noroom'] = clf.predict_proba(test_x)[:,1]
    testa = testa[['orderid', 'noroom']]
    file = './suball/'+str(i)+'.csv'
    testa.to_csv(file, index = None)
    test_a = pd.read_csv('../data/test/ord_testA.csv', encoding='gb2312')
    test_a = test_a[['orderid','room','arrival']]
    test_a = pd.merge(test_a, testa, on = 'orderid', how = 'left')
    file = './sub/'+str(i)+'qq.csv'
    print(file)
    test_a.to_csv(file, index = False)
    
    
    print('predict time : ', time.time() - start)
    start = time.time()








