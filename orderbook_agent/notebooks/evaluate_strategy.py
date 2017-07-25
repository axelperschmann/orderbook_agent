import joblib
from joblib import Parallel, delayed
import multiprocessing

from tqdm import tqdm, tqdm_notebook

import pandas as pd
import numpy as np
import gzip
import json
import math
import os
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

folder = 'trainedAgents/longterm_1611_1704_simulate_preceeding_trades_test'
# folder = 'trainedAgents/USDTETH'
agent_files = sorted([(f, folder) for f in os.listdir(folder) if f.endswith('.json')])
agent_files = [(f, folder) for (f, folder) in agent_files if 'shuffle' in f]
print(agent_files)
# agent_files.append( ('QTable_1611-1704_T4_I8_VolTime.json', 'trainedAgents/longterm_1611_1704_simulate_preceeding_trades_test') )
# agent_files = [('BT_Agent_samples36973.json', 'trainedAgents/longterm_apr_BT')]
print(agent_files)

agent_collection = {}
try:
    agent_collection
except NameError:
    agent_collection = {}
    
for elem, folder in agent_files:
    name = elem[:-5]
    print(name)
    agent_collection[name] = RLAgent_Base.load(agent_name=name, path=folder, ignore_samples=True)
    print(agent_collection[name])

data_may = pickle.load( open( '../cached_windows_60mins_V200/obs_2017-05_USDT_BTC_maxVol200.p', "rb" ) )
print(len(data_may))
# data_eth_jun = pickle.load( open( "/home/axel/notebooks/orderbook_agent/orderbook_agent/cached_windows_USDTETH/obs_2017-06_USDT_ETH_maxVol1000.p", "rb" ) )
# data_eth_jul = pickle.load( open( "/home/axel/notebooks/orderbook_agent/orderbook_agent/cached_windows_USDTETH/obs_2017-07_USDT_ETH_maxVol1000.p", "rb" ) )
# data = data_eth_jun + data_eth_jul[:256]
# data = data_eth_jul[256:]

baseline = list(agent_collection.keys())[0]
costs, slippage = evaluate(
    testdata=data,
    agents=agent_collection,
    baseline=baseline,
    evaluate_actions=[2, 4, 'MarketOrder'],
    verbose=False
)
slippage.to_csv('slippage_shuffle.csv')   # slippage_apr_manyVars3Bins_simulatedTrades, fixedMarketVar
#plot_evaluation_costs(slippage, hline="2", showfliers=False)