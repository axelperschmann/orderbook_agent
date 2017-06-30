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

# ## Preprocess data
# histfiles = [
#     "../../../../data/history/history_2016-11_USDT_BTC.csv",
# #     "../../../../data/history/history_2016-12_USDT_BTC.csv",
# #     "../../../../data/history/history_2017-01_USDT_BTC.csv",
# #     "../../../../data/history/history_2017-02_USDT_BTC.csv",
# ]
# # 
# hist = load_and_preprocess_historyfiles(histfiles)
# # 
# hist['future15_disc'] = pd.cut(hist.future15, bins=[-np.inf, -0.005, -0.001, 0.001, 0.005, np.inf], labels=False)
# hist['future30_disc'] = pd.cut(hist.future30, bins=[-np.inf, -0.005, -0.001, 0.001, 0.005, np.inf], labels=False)
# hist['future45_disc'] = pd.cut(hist.future45, bins=[-np.inf, -0.005, -0.001, 0.001, 0.005, np.inf], labels=False)
# hist['spread_disc'] = pd.cut(hist.spread, bins=[0, 1, 2, np.inf], labels=False)
# # display(hist.iloc[1021:1025,:])



# data = pickle.load( open( "cached_windows/tradingwindows_1611_USTD_BTC_20.p", "rb" ) )
# data = Parallel(n_jobs=num_cores, verbose=10)(delayed(add_features_to_orderbooks)(orderbooks=window, hist=hist, reset_features=True) for window in data[:])

## load data
data_nov = pickle.load( open( '../cached_windows_60mins_V200/obs_2016-11_USDT_BTC_maxVol200.p', "rb" ) )
print("data_nov", len(data_nov))
# data_nov = Parallel(n_jobs=num_cores, verbose=10)(delayed(add_features_to_orderbooks)(orderbooks=window, hist=hist) for window in data_nov[:])
# print(data_nov[0][0])
# 
data_dec = pickle.load( open( '../cached_windows_60mins/obs_2016-12_USDT_BTC_maxVol100.p', "rb" ) )
print("data_dec", len(data_dec))
# # data_dec = Parallel(n_jobs=num_cores, verbose=10)(delayed(add_features_to_orderbooks)(orderbooks=window, hist=hist) for window in data_dec[:])
# 
data_jan = pickle.load( open( '../cached_windows_60mins/obs_2017-01_USDT_BTC_maxVol100.p', "rb" ) )
print("data_jan", len(data_jan))
# # data_jan = Parallel(n_jobs=num_cores, verbose=10)(delayed(add_features_to_orderbooks)(orderbooks=window, hist=hist) for window in data_jan[:])
# 
data_feb = pickle.load( open( "../cached_windows_60mins/obs_2017-02_USDT_BTC_maxVol100.p", "rb" ) )
print("data_feb", len(data_feb))
# # data_feb = Parallel(n_jobs=num_cores, verbose=10)(delayed(add_features_to_orderbooks)(orderbooks=window, hist=hist) for window in data_feb[:])
# 
data_mar = pickle.load( open( "../cached_windows_60mins_V200/obs_2017-03_USDT_BTC_maxVol200.p", "rb" ) )
print("data_mar", len(data_mar))
# # data_mar = Parallel(n_jobs=num_cores, verbose=10)(delayed(add_features_to_orderbooks)(orderbooks=window, hist=hist) for window in data_mar[:])
# 
data_apr = pickle.load( open( "../cached_windows_60mins_V200/obs_2017-04_USDT_BTC_maxVol200.p", "rb" ) )
print("data_apr", len(data_mar))
# data_apr = Parallel(n_jobs=num_cores, verbose=10)(delayed(add_features_to_orderbooks)(orderbooks=window, hist=hist) for window in data_apr[:])

################
### SETTINGS ###
################
data = data_nov + data_dec + data_jan + data_feb + data_mar + data_apr

T=4
P=15
V=70000
consume='cash'
print("T: {}, P: {}, V: {}, consume: '{}'".format(T, P, V, consume))

# actions = np.linspace(-0.4, 1.0, num=15)
actions = range(-4,11)
print("Trading windows: {} (each one: {} minutes)".format(len(data), len(data[0])))


### QTable Agent
T=4
P=15
print('currBid')
agent = trainer_QTable(orderbooks=data, V=V, T=T, consume=consume, actions=[round(a, 2) for a in actions],
                    limit_base='currBid', vol_intervals=8,
                    period_length=P, agent_name='QTable_1611-1704_T4_I8_VolTime',
                    state_variables=['volume', 'time'], mode='backward')
agent.save(path="trainedAgents/longterm_1611_1704_currBid", overwrite=True)
