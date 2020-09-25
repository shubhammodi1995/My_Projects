#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from math import sqrt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from random import sample
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.cluster import SpectralClustering
import pandas_datareader as pdr
import datetime as dt
from matplotlib import pyplot as plt
import numpy as np
import sys
from IPython.display import clear_output
import os
from sklearn import cluster, preprocessing
from scipy.optimize import fsolve, minimize, basinhopping
import cvxopt as opt
from cvxopt import blas, solvers
import ipywidgets as widgets
import time
import bs4 as bs
import pickle
import requests
from sklearn.linear_model import LinearRegression
from guppy import hpy
import quandl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, validation_curve, TimeSeriesSplit
from sklearn.metrics import plot_roc_curve
import pickle


# In[2]:


pd.set_option('display.max_columns', None)


# # Downloading Data

# In[2]:


resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})
tickers = []
for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text.rstrip()
    tickers.append(ticker)


# In[4]:


all_data = pd.DataFrame()
test_data = pd.DataFrame()
no_data = []

for i in tickers:
    try:
        print(i)
        test_data = pdr.get_data_yahoo(i, start = dt.datetime(1990,1,1), end = dt.date.today())
        test_data['symbol'] = i
        all_data = all_data.append(test_data)
        clear_output(wait = True)
    except:
        no_data.append(i)


# # Importing Data

# In[5]:


#Count records for each company
counts = all_data.groupby('symbol')['Adj Close'].apply(lambda x: x.count())


# In[6]:


#Obtain companies with lower than 1000 records (3 years)
low_counts = counts[np.where(counts<1000)[0]].index.values


# In[7]:


all_data = all_data[~(all_data.symbol.isin(low_counts))]


# In[11]:


all_data.to_csv("sp500_data.csv")


# In[3]:


all_data = pd.read_csv("sp500_data.csv")


# In[4]:


all_data.index = all_data['Date']
all_data = all_data.drop(columns = ['Date'])


# # Creating Required Variables

# In[5]:


all_data['return'] = all_data.groupby('symbol')['Close'].pct_change()


# In[6]:


all_data['SMA_3'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 3).mean())


# In[7]:


all_data['SMA_15'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 15).mean())


# In[8]:


all_data['SMA_ratio'] = all_data['SMA_3'] / all_data['SMA_15']


# In[9]:


all_data['prev_close'] = all_data.groupby('symbol')['Close'].shift(1)


# In[10]:


all_data['TR'] = np.maximum((all_data['High'] - all_data['Low']), 
                     np.maximum(abs(all_data['High'] - all_data['prev_close']), 
                     abs(all_data['prev_close'] - all_data['Low'])))


# In[17]:


def Wilder(data, periods):
    Wilder = np.array([np.nan]*len(data))
    Wilder[periods] = data[0:(periods+1)].mean()
    for i in range(periods+1,len(data)):
        Wilder[i] = (Wilder[i-1]*(periods-1) + data[i])/periods
    return(Wilder)


# In[ ]:


for i in all_data['symbol'].unique():
    print(i)
    TR_data = all_data[all_data.symbol == i].copy()
    all_data.loc[all_data.symbol==i,'Wilder_TR_3'] = Wilder(TR_data['TR'], 3)
    all_data.loc[all_data.symbol==i,'Wilder_TR_15'] = Wilder(TR_data['TR'], 15)


# In[20]:


all_data.to_csv("sp500_till_WilderTR.csv")


# In[21]:


all_data['ATR_Ratio'] = all_data['Wilder_TR_3'] / all_data['Wilder_TR_15']


# In[22]:


all_data['SMA3_Volume'] = all_data.groupby('symbol')['Volume'].transform(lambda x: x.rolling(window = 3).mean())
all_data['SMA15_Volume'] = all_data.groupby('symbol')['Volume'].transform(lambda x: x.rolling(window = 15).mean())


# In[23]:


all_data['SMA_Volume_Ratio'] = all_data['SMA3_Volume']/all_data['SMA15_Volume']


# In[25]:


all_data['prev_high'] = all_data.groupby('symbol')['High'].shift(1)
all_data['prev_low'] = all_data.groupby('symbol')['Low'].shift(1)


# In[62]:


all_data['+DM'] = np.where(~np.isnan(all_data.prev_high),
                           np.where((all_data['High'] > all_data['prev_high']) & 
         (((all_data['High'] - all_data['prev_high']) > (all_data['prev_low'] - all_data['Low']))), 
                                                                  all_data['High'] - all_data['prev_high'], 
                                                                  0),np.nan)


# In[63]:


all_data['-DM'] = np.where(~np.isnan(all_data.prev_low),
                           np.where((all_data['prev_low'] > all_data['Low']) & 
         (((all_data['prev_low'] - all_data['Low']) > (all_data['High'] - all_data['prev_high']))), 
                                    all_data['prev_low'] - all_data['Low'], 
                                    0),np.nan)


# In[ ]:


for i in all_data['symbol'].unique():
    print(i)
    ADX_data = all_data[all_data.symbol == i].copy()
    all_data.loc[all_data.symbol==i,'+DM_3'] = Wilder(ADX_data['+DM'], 3)
    all_data.loc[all_data.symbol==i,'-DM_3'] = Wilder(ADX_data['-DM'], 3)
    all_data.loc[all_data.symbol==i,'+DM_15'] = Wilder(ADX_data['+DM'], 15)
    all_data.loc[all_data.symbol==i,'-DM_15'] = Wilder(ADX_data['-DM'], 15)


# In[71]:


all_data['+DI_3'] = (all_data['+DM_3']/all_data['Wilder_TR_3'])*100
all_data['-DI_3'] = (all_data['-DM_3']/all_data['Wilder_TR_3'])*100
all_data['+DI_15'] = (all_data['+DM_15']/all_data['Wilder_TR_15'])*100
all_data['-DI_15'] = (all_data['-DM_15']/all_data['Wilder_TR_15'])*100


# In[72]:


all_data['DX_3'] = (np.round(abs(all_data['+DI_3'] - all_data['-DI_3'])/(all_data['+DI_3'] + all_data['-DI_3']) * 100)) 


# In[73]:


all_data['DX_15'] = (np.round(abs(all_data['+DI_15'] - all_data['-DI_15'])/(all_data['+DI_15'] + all_data['-DI_15']) * 100)) 


# In[ ]:


for i in all_data['symbol'].unique():
    print(i)
    ADX_data = all_data[all_data.symbol == i].copy()
    all_data.loc[all_data.symbol==i,'ADX_3'] = Wilder(ADX_data['DX_3'], 3)
    all_data.loc[all_data.symbol==i,'ADX_15'] = Wilder(ADX_data['DX_15'], 15)


# In[77]:


all_data['Lowest_3D'] = all_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = 3).min())
all_data['High_3D'] = all_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window = 3).max())
all_data['Lowest_15D'] = all_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = 15).min())
all_data['High_15D'] = all_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window = 15).max())


# In[78]:


all_data['Stochastic_3'] = ((all_data['Close'] - all_data['Lowest_3D'])/(all_data['High_3D'] - all_data['Lowest_3D']))*100
all_data['Stochastic_15'] = ((all_data['Close'] - all_data['Lowest_15D'])/(all_data['High_15D'] - all_data['Lowest_15D']))*100


# In[79]:


all_data['Stochastic_Ratio'] = all_data['Stochastic_3']/all_data['Stochastic_15']


# In[80]:


all_data['Close_Low_200D'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 200).min())
all_data['Close_High_200D'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 200).max())


# In[81]:


all_data['Lowest_Price_Ratio'] = all_data['Close_Low_200D'] / all_data['Close']
all_data['Highest_Price_Ratio'] = all_data['Close'] / all_data['Close_High_200D']


# In[82]:


all_data['Diff'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.diff())

all_data['Up'] = all_data['Diff']
all_data.loc[(all_data['Up']<0), 'Up'] = 0

all_data['Down'] = all_data['Diff']
all_data.loc[(all_data['Down']>0), 'Down'] = 0 
all_data['Down'] = abs(all_data['Down'])

all_data['avg_3up'] = all_data.groupby('symbol')['Up'].transform(lambda x: x.rolling(window=3).mean())
all_data['avg_3down'] = all_data.groupby('symbol')['Down'].transform(lambda x: x.rolling(window=3).mean())

all_data['avg_15up'] = all_data.groupby('symbol')['Up'].transform(lambda x: x.rolling(window=15).mean())
all_data['avg_15down'] = all_data.groupby('symbol')['Down'].transform(lambda x: x.rolling(window=15).mean())

all_data['RS_3'] = all_data['avg_3up'] / all_data['avg_3down']
all_data['RS_15'] = all_data['avg_15up'] / all_data['avg_15down']

all_data['RSI_3'] = 100 - (100/(1+all_data['RS_3']))
all_data['RSI_15'] = 100 - (100/(1+all_data['RS_15']))


# In[83]:


all_data['RSI_ratio'] = all_data['RSI_3']/all_data['RSI_15']


# In[84]:


all_data['12Ewm'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
all_data['26Ewm'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())


# In[85]:


all_data['MACD'] = all_data['26Ewm'] - all_data['12Ewm']


# # Creating predicted variable

# In[86]:


all_data['Close_High_3D'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 3).max())


# In[87]:


all_data['Close_High_Shifted'] = all_data.groupby('symbol')['Close_High_3D'].transform(lambda x: x.shift(-3))


# In[88]:


all_data['Target'] = (all_data['Close_High_Shifted'] - all_data['Close'])/(all_data['Close']) * 100


# In[90]:


all_data['Target_Direction'] = np.where(all_data['Target']>0,1,0)


# In[91]:


all_data.to_csv("sp500_calculated.csv")


# # Entire Sample Data Preparation

# In[3]:


all_data = pd.read_csv("sp500_calculated.csv")
all_data.index = pd.DatetimeIndex(all_data.Date)
all_data = all_data.drop(columns= ['Date'])


# In[4]:


all_data = all_data.loc[all_data.symbol!='TT',:]


# # Clustering

# In[5]:


returns = all_data[['symbol','return']].copy()


# In[6]:


returns['Date'] = returns.index.copy()


# In[7]:


transposed = returns.pivot(index = 'Date', columns = 'symbol', values = 'return')


# In[7]:


from sklearn.mixture import GaussianMixture


# In[10]:


gmm = GaussianMixture(n_components = 60)


# In[11]:


gmm.fit(transposed.dropna().transpose())


# In[12]:


clusters = gmm.predict(transposed.dropna().transpose())


# In[14]:


clusters_df = pd.DataFrame({'Cluster':clusters,
                           'Companies':transposed.columns})


# In[15]:


clusters_df = clusters_df.sort_values(['Cluster']).reset_index(drop = True)


# In[17]:


clusters_df.to_csv("clusters.csv")


# In[13]:


clusters_df = pd.read_csv("clusters.csv", index_col = 0)


# # Training

# In[ ]:


for cluster_selected in clusters_df.Cluster.unique():
    
    print(f'The current cluster running is : {cluster_selected}')
    
    co_data = all_data[all_data.symbol.isin(clusters_df.loc[clusters_df.Cluster==cluster_selected,'Companies'].tolist())].copy()
    co_train = co_data[:'2019-01-01']
    co_train = co_train.dropna().copy()
    
    X_train = co_train.loc[:,['SMA_ratio','Wilder_TR_3','Wilder_TR_15','ATR_Ratio',
                       'ADX_3','ADX_15','SMA_Volume_Ratio','Stochastic_3','Stochastic_15','Stochastic_Ratio',
                      'Lowest_Price_Ratio','Highest_Price_Ratio','RSI_3','RSI_15','RSI_ratio','MACD']]

    Y_train = co_train.loc[:,['Target_Direction']]

    params = {'max_depth': [12, 15],
          'max_features': ['sqrt'],
          'min_samples_leaf': [10, 15, 20],
          'n_estimators': [150, 200, 250],
         'min_samples_split':[20, 25, 30]} #Using Validation Curves

    rf = RandomForestClassifier()

    time_series_split = TimeSeriesSplit(n_splits = 3)

    rf_cv = RandomizedSearchCV(rf, params, n_iter = 25, cv = time_series_split, n_jobs = -1, verbose = 20)

    rf_cv.fit(X_train, Y_train)
          
    file_loc = f'C:\\Users\\modis\\Desktop\\My Work\\Python\\Trading_Strategy_Summer_2020\\Pickle_Files\\Cluster_{cluster_selected}'    
    pickle.dump(rf_cv, open(file_loc,'wb'))
    
    


# # Trade

# In[3]:


all_data = pd.read_csv("sp500_calculated.csv")
all_data.index = pd.DatetimeIndex(all_data.Date)
all_data = all_data.drop(columns= ['Date'])

clusters_df = pd.read_csv("clusters.csv", index_col = 0)


# In[4]:


Start_Trade_Date = '2019-01-02'


# In[ ]:


data = all_data[Start_Trade_Date:]
del(all_data)


# In[69]:


dates = data.index.unique()


# In[81]:


trade_columns = []
for i in range(3):
    trade_columns += [f'Open_{i}', f'Close_{i}', f'symbol_{i}',f'Prediction_{i}', 
                                     f'Buy_{i}', f'Buy_Price_{i}', f'Position_Price_{i}',f'Daily_PL_{i}',
                      f'Value_Open_{i}', f'Value_Close_{i}', 
                                     f'Sell_{i}',f'Sell_Price_{i}',f'Net_Position_{i}',f'Cum_Ret_{i}']
trade_columns+=['Final_Net_Position','Tot_Value_Open','Tot_Value_Close','Tot_Cum_Ret']


# In[82]:


co_trades = pd.DataFrame(index = dates, 
                          columns = trade_columns)

co_trades.loc[co_trades.index[0],'Tot_Cum_Ret'] = 1

co_trades.loc[co_trades.index[0],[f'Net_Position_{i}' for i in range(3)]] = 0

co_trades.loc[co_trades.index[0],'Final_Net_Position'] = 0


# # Get Prediction for Next Day

# In[ ]:


for i in range(len(dates)-1):
    
    print(i)
    
    ##Get Prediction for Tomorrow##
    date = dates[i]
    print(date)
    
    if co_trades.loc[dates[i],'Final_Net_Position'] == 0:
        pred_for_tomorrow = pd.DataFrame({'company':[],
                                 'prediction':[],
                                 'Date':[]})
    
    
        for cluster_selected in clusters_df.Cluster.unique():
            rf_cv =  pickle.load(open(os.getcwd() + f'\\Pickle_Files\\Cluster_{cluster_selected}', 'rb'))
            best_rf = rf_cv.best_estimator_
            cluster_data = data.loc[data.symbol.isin(clusters_df.loc[clusters_df.Cluster==cluster_selected,'Companies'].tolist())].loc[[date]].copy()
            cluster_data = cluster_data.dropna()
            if (cluster_data.shape[0]>0):
                X_test = cluster_data[['SMA_ratio','Wilder_TR_3','Wilder_TR_15','ATR_Ratio',
                               'ADX_3','ADX_15','SMA_Volume_Ratio','Stochastic_3','Stochastic_15','Stochastic_Ratio',
                              'Lowest_Price_Ratio','Highest_Price_Ratio','RSI_3','RSI_15','RSI_ratio','MACD']]

                pred_for_tomorrow = pred_for_tomorrow.append(pd.DataFrame({'company':cluster_data['symbol'],
                                                                           'prediction':rf_cv.predict_proba(X_test)[:,1],
                                                                          'Date':cluster_data.index}), ignore_index = True)
            else:
                continue
        pred_for_tomorrow = pred_for_tomorrow.sort_values(by = ['prediction'], ascending= False).reset_index(drop = True)

    i += 1
    
    
    ##Trade on the Prediction##
    for j in range(3):
        co_trades.loc[dates[i],f'Open_{j}'] = data[data.symbol == pred_for_tomorrow.iloc[j].company].loc[dates[i]]['Open']
        co_trades.loc[dates[i],f'Close_{j}'] = data[data.symbol == pred_for_tomorrow.iloc[j].company].loc[dates[i]]['Close']
        co_trades.loc[dates[i],f'Prediction_{j}'] = 1
        co_trades.loc[dates[i],f'symbol_{j}'] = pred_for_tomorrow.iloc[j].company
        
    if co_trades.loc[dates[i-1],'Final_Net_Position'] == 0:
        
        companies = range(3)
        
        co_trades.loc[dates[i],['Buy_' + str(k) for k in companies]] = 1

        co_trades.loc[dates[i],['Buy_Price_' + str(k) for k in companies]] = np.array(co_trades.loc[dates[i],['Open_' + str(k) for k in companies]])*np.array(co_trades.loc[dates[i],['Buy_' + str(k) for k in companies]])

        co_trades.loc[dates[i],['Net_Position_' + str(k) for k in companies]] = 1
        
        co_trades.loc[dates[i],['Value_Open_' + str(k) for k in companies]] = 100
        
        co_trades.loc[dates[i],'Tot_Value_Open'] = sum(co_trades.loc[dates[i],['Value_Open_' + str(k) for k in companies]])

        co_trades.loc[dates[i],['Position_Price_' + str(k) for k in companies]] = np.array(co_trades.loc[dates[i],['Open_' + str(k) for k in companies]])
        
        position_day_count = 1

        co_trades.loc[dates[i],'Day_of_Trade'] = position_day_count
        
        co_trades.loc[dates[i],['Daily_PL_' + str(k) for k in companies]] = ((np.array(co_trades.loc[dates[i],['Close_' + str(k) for k in companies]])-
                                                     np.array(co_trades.loc[dates[i],['Open_' + str(k) for k in companies]]))/np.array(co_trades.loc[dates[i],['Open_' + str(k) for k in companies]])).tolist()
        
        co_trades.loc[dates[i],['Value_Close_' + str(k) for k in companies]] = np.array(co_trades.loc[dates[i],['Value_Open_' + str(k) for k in companies]]) * np.array(1 + co_trades.loc[dates[i],['Daily_PL_' + str(k) for k in companies]])
        
        co_trades.loc[dates[i],['Cum_Ret_' + str(k) for k in companies]] = 1 + np.array(co_trades.loc[dates[i],['Daily_PL_' + str(k) for k in companies]])
        
        co_trades.loc[dates[i],'Tot_Value_Close'] = sum(co_trades.loc[dates[i],['Value_Close_' + str(k) for k in companies]])

        co_trades.loc[dates[i],'Tot_Cum_Ret'] = co_trades.loc[dates[i-1],'Tot_Cum_Ret']*(co_trades.loc[dates[i],'Tot_Value_Close'] / co_trades.loc[dates[i],'Tot_Value_Open'])

        co_trades.loc[dates[i],'Final_Net_Position'] = 1
    
    else:
        
        companies = np.where(np.array(co_trades.loc[dates[i-1],['Net_Position_' + str(k) for k in range(3)]].tolist())==1)[0].tolist()
    
        co_trades.loc[dates[i],['Position_Price_' + str(k) for k in companies]] = np.array(co_trades.loc[dates[i-1],['Position_Price_' + str(k) for k in companies]])

        co_trades.loc[dates[i],'Day_of_Trade'] = position_day_count

        co_trades.loc[dates[i],['Net_Position_' + str(k) for k in companies]] = 1
        
        co_trades.loc[dates[i],['Value_Open_' + str(k) for k in companies]] = np.array(co_trades.loc[dates[i-1],['Value_Close_' + str(k) for k in companies]])
                
        co_trades.loc[dates[i],'Tot_Value_Open'] = co_trades.loc[dates[i-1],'Tot_Value_Close']

        co_trades.loc[dates[i],['Daily_PL_' + str(k) for k in companies]] = ((np.array(co_trades.loc[dates[i],['Close_' + str(k) for k in companies]])-
                                                     np.array(co_trades.loc[dates[i],['Open_' + str(k) for k in companies]]))/np.array(co_trades.loc[dates[i],['Open_' + str(k) for k in companies]])).tolist()
        
        co_trades.loc[dates[i],['Value_Close_' + str(k) for k in companies]] = np.array(co_trades.loc[dates[i],['Value_Open_' + str(k) for k in companies]]) * np.array(1 + co_trades.loc[dates[i],['Daily_PL_' + str(k) for k in companies]])
         
        co_trades.loc[dates[i],['Cum_Ret_' + str(k) for k in companies]] = np.array(co_trades.loc[dates[i-1],['Cum_Ret_' + str(k) for k in companies]])*(1 + np.array(co_trades.loc[dates[i],['Daily_PL_' + str(k) for k in companies]]))   
            
        co_trades.loc[dates[i],'Tot_Value_Close'] = sum(co_trades.loc[dates[i],['Value_Close_' + str(k) for k in companies]])

        co_trades.loc[dates[i],'Tot_Cum_Ret'] = co_trades.loc[dates[i-1],'Tot_Cum_Ret']*(co_trades.loc[dates[i],'Tot_Value_Close'] / co_trades.loc[dates[i],'Tot_Value_Open'])
    
        co_trades.loc[dates[i],'Final_Net_Position'] = 1
        
    
    if ((position_day_count == 15)):

        co_trades.loc[dates[i],['Net_Position_' + str(k) for k in companies]] = 0

        co_trades.loc[dates[i],['Sell_' + str(k) for k in companies]] = 1

        co_trades.loc[dates[i],['Sell_Price_' + str(k) for k in companies]] = np.array(co_trades.loc[dates[i],['Close_' + str(k) for k in companies]])

        position_day_count = 0
    
    co_trades.loc[dates[i],'Final_Net_Position'] = np.max(co_trades.loc[dates[i],['Net_Position_' + str(k) for k in companies]])
        
    position_day_count += 1


# In[ ]:




