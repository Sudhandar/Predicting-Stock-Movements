#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 12:08:31 2021

@author: sudhandar
"""

import pandas as pd
from datetime import timedelta, datetime
import re
import numpy as np
from sklearn.metrics import matthews_corrcoef

def get_focus_dates(date, lag):
    start_date = date - timedelta(days = lag)
    end_date = date
    date_range = pd.DataFrame(pd.date_range(start=start_date, end=end_date),columns = ['focus_dates'])
    date_range['stock_date'] = date
    date_range['start_date'] = 0
    date_range['end_date'] = 0
    date_range['start_date'][0] = 1
    date_range['end_date'][int(date_range.shape[0])-1] = 1

    return date_range

def get_date_range_df(stock,lag):
    date_df = pd.DataFrame()
    stock_dates = stock[['datetime']].drop_duplicates()
    for date in stock_dates['datetime']:
        date_range = get_focus_dates(date, lag)
        date_df = date_df.append(date_range)
    return date_df

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


tweets = pd.read_csv('tweets_combined.csv',lineterminator='\n')
stock = pd.read_csv('stock_market_filtered_dates.csv')
stock['datetime'] = pd.to_datetime(stock['date'])
stock = stock.drop_duplicates()
stock['market_binary'].value_counts()
tweets['datetime'] = pd.to_datetime(tweets['time'])
tweets['hour']= tweets.datetime.dt.hour
tweets_filtered = tweets[tweets['datetime']>"01/01/2014"]
tweets_filtered = tweets_filtered[tweets_filtered['datetime']<'01/01/2016']


lag = 5
combined_output = pd.DataFrame()
stock_list = list(set(stock['stock_symbol'].values.tolist()))
for stock_symbol in stock_list:
    stock_df = stock[stock['stock_symbol']==stock_symbol]
    tweets_df = tweets[tweets['stock_symbol']==stock_symbol]
    date_range = get_date_range_df(stock_df,lag)
    date_range['date'] = date_range['focus_dates'].apply(lambda x: x.date())
    tweets_df['date'] = tweets_df['datetime'].apply(lambda x: x.date())
    selected_tweets = pd.merge(date_range,tweets_df,on='date')
    combined_output = combined_output.append(selected_tweets)
    start_dates = combined_output[combined_output['start_date']==1]
    start_dates = start_dates[start_dates['hour']>15]
    end_dates = combined_output[combined_output['end_date']==1]
    end_dates = end_dates[end_dates['hour']<9]
    combined_output = combined_output[(combined_output['start_date']==0)&(combined_output['end_date']==0)]
    combined_output = combined_output.append(start_dates)
    combined_output = combined_output.append(end_dates)

combined_output['datetime'] = pd.to_datetime(combined_output['stock_date'])
combined_output_merged = pd.merge(stock,combined_output,on=['datetime','stock_symbol'],how='inner')
combined_output_merged = combined_output_merged[['stock_date', 'movement percent', 'stock_symbol', 'market_binary','tweet_id']].drop_duplicates()
combined_output_merged.columns = ['stock_date', 'movement_percent', 'stock_symbol', 'market_binary','tweet_id']
combined_output_merged = combined_output_merged.sort_values(['stock_date','stock_symbol'],ascending = [True,True])


train = combined_output_merged[combined_output_merged['stock_date']<'2015-08-01']
val = combined_output_merged[(combined_output_merged['stock_date']>='2015-08-01')&(combined_output_merged['stock_date']<'2015-10-01')]
test = combined_output_merged[combined_output_merged['stock_date']>='2015-10-01']


sentiment_output = pd.read_csv('tweets_combined_sentiment_output.csv',encoding='latin-1',lineterminator='\n')
sentiment_output = sentiment_output[['tweet_id','predictions']].drop_duplicates()
test = test[['stock_date','stock_symbol','movement_percent', 'market_binary','tweet_id']].drop_duplicates()
combined_output_test = pd.merge(test,sentiment_output,on='tweet_id',how='inner')
combined_output_test['pred'] = combined_output_test.groupby(['stock_date','stock_symbol'])['predictions'].transform(np.mean)
combined_output_test = combined_output_test[['stock_date','stock_symbol','movement_percent', 'market_binary','pred']].drop_duplicates()
combined_output_test.loc[combined_output_test['pred']<=0.5,'pred_label'] = 0
combined_output_test.loc[combined_output_test['pred']>0.5,'pred_label'] = 1
accuracy = np.sum(combined_output_test['pred_label'] == combined_output_test['market_binary']) / len(combined_output_test['market_binary'])
mcc = matthews_corrcoef(combined_output_test['market_binary'], combined_output_test['pred_label'])

print('Accuracy:{0:.3f}'.format(accuracy*100))
print('MCC:{0:.3f}'.format(mcc))


