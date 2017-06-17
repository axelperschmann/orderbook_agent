from joblib import Parallel, delayed
import multiprocessing

from tqdm import tqdm, tqdm_notebook
from IPython.display import display
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import os
import sys
sys.path.append('..')
from helper.orderbook_trader import OrderbookTradingSimulator
from helper.manage_orderbooks import *
from helper.evaluation import evaluate, plot_evaluation_costs
from helper.general_helpers import (
    add_features_to_orderbooks,
    load_and_preprocess_historyfiles,
    discretize_hist_feature,
    addMarketFeatures_toSamples)

from agents.RL_Agent_Base import RLAgent_Base
from agents.BatchTree_Agent import RLAgent_BatchTree
from agents.QTable_Agent import QTable_Agent


### Load source agent
folder = 'trainedAgents/longterm'
agent_name = 'QTable_1612-1702_T4_I8'

agent_source = RLAgent_Base.load(agent_name=agent_name, path=folder)
print(agent_source)

### Dublicate Agent, modify samples
new_feature = 'marketPrice_buy_worst'
new_agent = agent_source.copy(new_name='{}_{}_obFixed'.format(agent_source.agent_name, new_feature))
print("New agent: {}".format(new_agent.agent_name))
new_agent.state_variables = ['volume', 'time', '{}_disc'.format(new_feature)]

hist = pd.read_csv('ob_features70000_1611_1702_obFixed.csv', index_col=0, parse_dates=[0])
hist = discretize_hist_feature(hist, feature=new_feature, test_start_date='2017-03-01')

new_agent.samples = addMarketFeatures_toSamples(
    samples=new_agent.samples, hist=hist,
    market_features=['{}_disc'.format(new_feature)]
)

print("Learn from samples:", new_agent.samples.shape)
new_agent.learn_fromSamples(verbose=True)

new_agent.save(path='trainedAgents/longterm')

#sharecount_imbalance, maxSlip_imbalance, marketPrice_spread, marketPrice_buy_worst