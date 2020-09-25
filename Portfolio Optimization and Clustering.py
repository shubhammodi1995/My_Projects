#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
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
from sklearn.datasets.samples_generator import make_blobs
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


# # Import Data

# In[3]:


stocks_ret = pd.read_csv("Stocks Return.csv", index_col = 0)
stocks_price = pd.read_csv("Stocks Price.csv", index_col = 0)

data_return = stocks_ret
data_price = stocks_price

data_corr = pd.DataFrame(np.corrcoef(data_return))
data_corr.index = data_return.index


# # Final Code

# In[4]:


def Optimization_Tool(data):

    ############# Get required values from user #####################
    toggle_to_cluster = widgets.ToggleButtons( 
        options=['Clustered', 'Actual'], description='Portfolio', disabled=False, button_style='', value = 'Clustered')

    toggle_clustering = widgets.ToggleButtons(
        options=['Returns', 'Price', 'Correlations'], description='Clustering', button_style='', disabled = False, value = 'Returns')

    clusters = widgets.IntSlider(min=10, max=40, value=25, description='Clusters')

    toggle_weighting = widgets.ToggleButtons(
        options=['Sharpe Ratio', 'Equal', 'Highest SR'], 
        description='Weighting', button_style='', disabled = False, value = 'Sharpe Ratio')

    toggle_optimization = widgets.ToggleButtons(
        options=['Random Portfolios', 'Sharpe Ratio', 'SR Expanding', 'SR Rolling'],
        description='Optimization', disabled=False, button_style='', value = 'Sharpe Ratio'
    )

    toggle_shorting = widgets.ToggleButtons(
        options=['Yes', 'No'], description='Shorting?', disabled=False, button_style='', value = 'Yes'
    )

    button = widgets.Button(description="Submit")

    def on_button_clicked(b):
        #toggle_to_cluster.disabled = True
        #toggle_clustering.disabled = True
        #toggle_optimization.disabled = True
        #toggle_shorting.disabled = True
        #clusters.disabled = True
        #toggle_weighting.disabled = True
        
        portfolio_type = toggle_to_cluster.value
        clustering_method = toggle_clustering.value
        weighting_method = toggle_weighting.value
        number_of_clusters = clusters.value
        to_short = toggle_shorting.value
        optimization = toggle_optimization.value
            
        OptimalWeights(data, portfolio_type, clustering_method, weighting_method, number_of_clusters, to_short, optimization)
          

    button.on_click(on_button_clicked)

    def to_cluster(change):
        toggle_clustering.disabled = True if change.new == 'Actual' else False
        clusters.disabled = True if change.new == 'Actual' else False
        toggle_weighting.disabled = True if change.new == 'Actual' else False

    toggle_to_cluster.observe(to_cluster, names='value')

    display(toggle_to_cluster, toggle_clustering, clusters, toggle_optimization, toggle_shorting, toggle_weighting, button)


# In[5]:


def OptimalWeights(stocks_price, portfolio_type, clustering_method, weighting_method, 
                   number_of_clusters, to_short, optimization, risk_free_rate = 0.02):
    
    global optimal_weights, shorting, stocks_ret_outsample,     Cluster_portfolio, returns_insample, returns_outsample,     MVE_weights, in_sample_returns, out_sample_returns, stocks_ret_outsample
    
    ################# Some fancy loading ###################
    progress = widgets.FloatProgress(value=0, min=0, max=100, step=1, description='Loading:', bar_style='info',
                                     orientation='horizontal')
    display(progress)
    
    ############# Creating Required Data #############
    progress.value = 20
    
    start_train = '2000-01-01'
    end_train = '2017-12-31'
    
    start_test = '2018-01-01'
    end_test = '2020-06-31'
    
    stocks_price_full = stocks_price.loc[start_train:end_test,:]
    stocks_ret_full = stocks_price_full.pct_change().dropna()
    
    ############# Declaring Required Variables ###############

    
    stocks_price_insample = stocks_price_full.loc[start_train:end_train,:]
    stocks_ret_insample = stocks_ret_full.loc[start_train:end_train,:]

    stocks_ret_outsample = stocks_ret_full.loc[start_test:end_test,:]
    
    shorting = True if to_short == 'Yes' else False 
    
    stocks_corr_insample = pd.DataFrame(np.corrcoef(stocks_ret_insample.T))
    stocks_corr_insample.index = stocks_ret_insample.columns
    stocks_corr_insample.columns = stocks_ret_insample.columns
    
    ################ Create Clusters ################

    def Clustering_Data(data, number_of_clusters):
        data_1 = data
        data_2 = pd.DataFrame(preprocessing.scale(data_1))
        data_2.columns = data_1.columns
        data_2.index = data_2.index
        data_3 = np.transpose(data_2)

        NumOfCluster = number_of_clusters
        Hcluster = cluster.AgglomerativeClustering(NumOfCluster)
        Hcluster.fit(data_3)
        label = Hcluster.labels_
        VarName = data_3.index.values.tolist()
        ClusterResult = dict(zip(VarName, label))
        ClusterResult_1 = sorted(ClusterResult.items(), key=lambda item:item[1]) 
        ClusterResult_2 = pd.DataFrame(ClusterResult_1)
        ClusterResult_2.columns = ['VarName', 'Cluster']
        
        return(ClusterResult_2)

     ################ Create Cluster Return using required Weights  ################

    def Weighted_Portfolios(return_data_full, return_data_insample, weighting_method, cluster_data):
        data_1 = return_data_insample
        means = np.mean(data_1,axis=0)
        sd = np.std(data_1,axis=0)
        sratio = pd.DataFrame(means/sd)
        ClusterResult_2 = cluster_data.sort_values("VarName").reset_index(drop=True)
        sratio = sratio.sort_index()
        ClusterResult_2["SharpeRatio"] = sratio[0].reset_index(drop=True)
        ClusterResult_2["EqualWeighted"] = 1
        ClusterResult_2["EqualWeighted"] = ClusterResult_2.groupby('Cluster')['EqualWeighted'].transform(lambda x: 1/x.size)
        ClusterResult_2['HighestSharpeRatio'] = ClusterResult_2.groupby('Cluster')['SharpeRatio'].transform(lambda x: x.max())
        ClusterResult_2['HighestSharpeRatioWeights'] = ClusterResult_2.apply(lambda x: 1 if x['SharpeRatio'] == x['HighestSharpeRatio'] else 0, axis = 1)
        #Check if any cluster has two or more companies with the sharpe ratio and adjust weights if so.
        ClusterResult_2['HighestSharpeRatioWeights'] = ClusterResult_2.groupby('Cluster')['HighestSharpeRatioWeights'].transform(lambda x: x/x.sum())
        ClusterResult_2['SharpeRatioWeight'] = ClusterResult_2.groupby('Cluster')['SharpeRatio'].transform(lambda x: x/x.sum())
        ClusterResult_2.index = ClusterResult_2.VarName 
        
        global ClusterResult 
        ClusterResult = ClusterResult_2
        
        try:
            if weighting_method == 'Sharpe Ratio':
                weighting = 'SharpeRatioWeight'
            elif weighting_method == 'Equal':
                weighting = 'EqualWeighted' 
            elif weighting_method == 'Highest SR':
                weighting = 'HighestSharpeRatio' 
            else:
                raise SyntaxError()
        except SyntaxError:
            print("Please select the right parameters")

        portfolio = pd.DataFrame()
        for i in range(0, number_of_clusters):
            subcluster = ClusterResult_2.loc[ClusterResult_2['Cluster'] == i]['VarName']
            portfolio["Cluster_" + str(i+1)] = np.dot(return_data_full[subcluster], ClusterResult_2[ClusterResult_2['VarName'].isin(subcluster.values)][weighting])

        portfolio.index = return_data_full.index
        
       
        return(portfolio)

    ############### Long-Short Weights Adjustment ##################

    def long_short_weights(x, values):
            values = values*x
            return((np.abs(values[values<0]).sum())/2 + values[values>0].sum() - 1)


    ############### Random Portfolios ######################

    def calculate_returns(weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns*weights) *252
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return std, returns

    def generate_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
        results = np.zeros((3,num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.randn(mean_returns.shape[0]) if shorting else np.random.random(mean_returns.shape[0])
            weights = weights * fsolve(long_short_weights, 0, args = (weights))
            weights_all.append(weights)
            portfolio_std_dev, portfolio_return = calculate_returns(weights, mean_returns, cov_matrix)
            results[0,i] = portfolio_std_dev
            results[1,i] = portfolio_return
            results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        return results, weights_all

    def optimal_weights(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
        results, weights = generate_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)

        max_sharpe_idx = np.argmax(results[2])
        sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]

        MVE_weights = pd.DataFrame(weights[max_sharpe_idx],index=data.columns,columns=['allocation'])
        MVE_weights = max_sharpe_weights.T

        return(MVE_weights)


    ############### Sharpe Ratio Optimization ###############
    
    def Max_SR(weights, MVE_data, optimization):
        returns = pd.DataFrame({'vals':np.array(np.dot(MVE_data,weights))},index = MVE_data.index)
        max_val = 0
        if optimization == 'SR Expanding':
            returns['expanding_sd'] = returns.vals.expanding().std()
            returns['expanding_mean_ret'] = returns.vals.expanding().mean()
            SR = returns.loc[start_train:end_train,'expanding_mean_ret'] / returns.loc[start_train:end_train,'expanding_sd']
            max_val = SR.mean()*np.sqrt(252)
        elif optimization == 'SR Rolling':
            returns['window_sd'] = returns.vals.rolling(window = 250).std()
            returns['window_returns'] = returns.vals.rolling(window = 250).mean()
            SR = returns.loc[start_train:end_train,'window_returns'] / returns.loc[start_train:end_train,'window_sd']
            max_val = SR.mean()*np.sqrt(252)
        elif optimization == 'Sharpe Ratio':
            SR =  returns.loc[start_train:end_train,'vals'].mean()/returns.loc[start_train:end_train,'vals'].std() * np.sqrt(252)
            max_val = SR
        return(-max_val)

    def portfolio_returns(MVE_data, optimization, shorting):
        initial_weights = [1/MVE_data.shape[1]]*MVE_data.shape[1]
        if shorting:
            cons = ({'type':'eq','fun': lambda x: np.sum(x)-1})
            bnds = [(None,None)]*MVE_data.shape[1]
        else:
            cons = ({'type':'eq','fun': lambda x: np.sum(x)-1})
            bnds = [(0,None)]*MVE_data.shape[1]

        #Currently using L-BFGS-B optimization
        MVE_weights = minimize(Max_SR, initial_weights, args = (MVE_data, optimization), bounds = bnds, constraints = cons).x
        progress.value = 50
        MVE_weights = MVE_weights * fsolve(long_short_weights, 0, args = (MVE_weights))
        progress.value = 60
        return(MVE_weights)


    ############## Calculating the most optimal weights #################
    
    if portfolio_type == 'Clustered':
        if clustering_method == 'Returns':
            cluster_data = stocks_ret_insample
        elif clustering_method == 'Price':
            cluster_data = stocks_price_insample
        else: cluster_data = stocks_corr_insample
        
        progress.value = 30
        
        Cluster_portfolio = Clustering_Data(cluster_data, number_of_clusters)
        progress.value = 40
        
        returns = Weighted_Portfolios(stocks_ret_full, stocks_ret_insample, weighting_method, Cluster_portfolio)
        returns_insample = returns.loc[start_train:end_train,:]
        returns_outsample = returns.loc[start_test:end_test,:]
        
        MVE_weights = portfolio_returns(returns, optimization, shorting)
        
    if portfolio_type == 'Actual':
        
        progress.value = 30
        returns_insample = stocks_ret_insample
        returns_outsample = stocks_ret_outsample
        MVE_weights = portfolio_returns(returns_insample, optimization, shorting)
    
    progress.value = 80
    in_sample_returns = np.dot(returns_insample, MVE_weights)
    
    fig, ax = plt.subplots(1,1, figsize = (20,10))
    ax.plot(pd.to_datetime(stocks_ret_insample.index.values,format = '%Y-%m-%d'), (in_sample_returns+1).cumprod())
    ax.title.set_text('In Sample Cumulative Returns')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Returns')
    plt.show()
    
    out_sample_returns = np.dot(returns_outsample, MVE_weights)
    
    fig, ax = plt.subplots(1,1, figsize = (20,10))
    ax.plot(pd.to_datetime(stocks_ret_outsample.index.values,format = '%Y-%m-%d'), (out_sample_returns+1).cumprod())
    ax.title.set_text('Out Sample Cumulative Returns')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Returns')
    plt.show()
    
    progress.value = 100
    print("Your optimal weights are: \n", MVE_weights)
    
    
    
    
        
    


# In[ ]:


Optimization_Tool(stocks_price.T)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




