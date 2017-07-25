import joblib
from joblib import Parallel, delayed
import multiprocessing

from tqdm import tqdm, tqdm_notebook

import pandas as pd
import numpy as np
import gzip
import json
import math
from datetime import datetime
import seaborn as sns
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display
import pickle
from random import shuffle

import sys
sys.path.append('..')
from helper.rl_framework import *
from helper.orderbook_container import OrderbookContainer
from helper.manage_orderbooks import *
from helper.orderbook_trader import *
from helper.evaluation import evaluate, plot_evaluation_costs
from helper.general_helpers import add_features_to_orderbooks, load_and_preprocess_historyfiles

from agents.RL_Agent_Base import RLAgent_Base
from agents.BatchTree_Agent import RLAgent_BatchTree
from agents.QTable_Agent import QTable_Agent
from Runs.train_agents import trainer_BatchTree, trainer_QTable

num_cores = multiprocessing.cpu_count()


# load data
data_nov = pickle.load( open( '../cached_windows_60mins_V200/obs_2016-11_USDT_BTC_maxVol200.p', "rb" ) )
print("data_nov", len(data_nov))
# data_nov = Parallel(n_jobs=num_cores, verbose=10)(delayed(add_features_to_orderbooks)(orderbooks=window, hist=hist) for window in data_nov[:])
# print(data_nov[0][0])

data_dec = pickle.load( open( '../cached_windows_60mins/obs_2016-12_USDT_BTC_maxVol100.p', "rb" ) )
print("data_dec", len(data_dec))
# # # data_dec = Parallel(n_jobs=num_cores, verbose=10)(delayed(add_features_to_orderbooks)(orderbooks=window, hist=hist) for window in data_dec[:])
# # 
data_jan = pickle.load( open( '../cached_windows_60mins/obs_2017-01_USDT_BTC_maxVol100.p', "rb" ) )
print("data_jan", len(data_jan))
# # # data_jan = Parallel(n_jobs=num_cores, verbose=10)(delayed(add_features_to_orderbooks)(orderbooks=window, hist=hist) for window in data_jan[:])
# # 
data_feb = pickle.load( open( "../cached_windows_60mins/obs_2017-02_USDT_BTC_maxVol100.p", "rb" ) )
print("data_feb", len(data_feb))
# # # data_feb = Parallel(n_jobs=num_cores, verbose=10)(delayed(add_features_to_orderbooks)(orderbooks=window, hist=hist) for window in data_feb[:])
# # 
data_mar = pickle.load( open( "../cached_windows_60mins_V200/obs_2017-03_USDT_BTC_maxVol200.p", "rb" ) )
print("data_mar", len(data_mar))
# # # data_mar = Parallel(n_jobs=num_cores, verbose=10)(delayed(add_features_to_orderbooks)(orderbooks=window, hist=hist) for window in data_mar[:])
# 
data_apr = pickle.load( open( "../cached_windows_60mins_V200/obs_2017-04_USDT_BTC_maxVol200.p", "rb" ) )
print("data_apr", len(data_apr))
# # data_apr = Parallel(n_jobs=num_cores, verbose=10)(delayed(add_features_to_orderbooks)(orderbooks=window, hist=hist) for window in data_apr[:])

data_may = pickle.load( open( "../cached_windows_60mins_V200/obs_2017-05_USDT_BTC_maxVol200.p", "rb" ) )
print("data_may", len(data_may))
# # data_apr = Parallel(n_jobs=num_cores, verbose=10)(delayed(add_features_to_orderbooks)(orderbooks=window, hist=hist) for window in data_apr[:])

################
### SETTINGS ###
################
data = data_nov + data_dec + data_jan + data_feb + data_mar + data_apr + data_may[50:52] + data_may[100:102] + data_may[200:205] + data_may[300:315] + data_may[400:405]

# data_eth_jun = pickle.load( open( "/home/axel/notebooks/orderbook_agent/orderbook_agent/cached_windows_USDTETH/obs_2017-06_USDT_ETH_maxVol1000.p", "rb" ) )
# data_eth_jul = pickle.load( open( "/home/axel/notebooks/orderbook_agent/orderbook_agent/cached_windows_USDTETH/obs_2017-07_USDT_ETH_maxVol1000.p", "rb" ) )
# data = data_eth_jun + data_eth_jul
# print("len(data)", len(data))
data = data[:int(len(data)*0.8)]

shuffle(data)
print("len(data)", len(data))

T=4
P=15
V=70000
consume='cash'
state_variables = ['volume', 'time', 'level2data']
print("T: {}, P: {}, V: {}, consume: '{}'".format(T, P, V, consume))

# actions = np.linspace(-0.4, 1.0, num=15)
actions = range(-4,11)
print("Trading windows: {} (each one: {} minutes)".format(len(data), len(data[0])))


print("BT-Agent")
### BT Agent
agent = trainer_BatchTree(orderbooks=data[:], V=V, T=T, consume=consume, actions=actions,
                    lim_stepsize=0.1, limit_base='currAsk',
                    period_length=P, epochs=60, agent_name='BT_Agent_shuffle',
                    random_start=False, state_variables=state_variables, 
                    mode='forward', retraining=24, savepath='trainedAgents/USDTETH')


agent.save(path="trainedAgents/USDTETH", overwrite=True)
