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
folder = 'trainedAgents/longterm_1611_1704_currBid'
agent_name = 'QTable_1611-1704_T4_I8_VolTime'

# folder = 'trainedAgents'
# agent_name = 'test_Dec24_VolTime'

agent_source = RLAgent_Base.load(agent_name=agent_name, path=folder)
hist = pd.read_csv('ob_features70000_1611_1705.csv', index_col=0, parse_dates=[0])
print(agent_source)


def retrain_and_save_agent(agent_source, hist, new_feature, test_start_date, bins, outputpath, verbose=True):
	print("outputpath", outputpath)
	try:
		hist = discretize_hist_feature(hist, feature=new_feature, test_start_date=test_start_date, bins=bins)
	except ValueError:
		print("not successful:", new_feature)
		return
	new_agent = agent_source.copy(new_name='{}_{}'.format(agent_source.agent_name, new_feature))
	new_agent.state_variables = ['volume', 'time', '{}_disc{}'.format(new_feature, bins)]

	if verbose:
		print("New agent: {}".format(new_agent.agent_name))
		print(new_agent)

	new_agent.samples = addMarketFeatures_toSamples(
    	samples=new_agent.samples, hist=hist,
    	market_features=['{}_disc{}'.format(new_feature, bins)]
	)
	if verbose:
		print("Learn from samples:", new_agent.samples.shape)
	new_agent.learn_fromSamples(verbose=True)

	new_agent.save(path=outputpath)

num_cores = multiprocessing.cpu_count()
features = ['ask', 'bid', 'center_orig', 'future_center15',
       'future_center5', 'future_center60', 'marketPrice_buy_worst',
       'marketPrice_sell_worst', 'marketPrice_spread', 'maxSlip_buy',
       'maxSlip_imbalance', 'maxSlip_sell', 'ob_direction', 'sharecount_buy',
       'sharecount_imbalance', 'sharecount_sell', 'sharecount_spread',
       'spread']
features = ['marketPrice_buy_worst', 'marketPrice_spread', 'maxSlip_buy', 'spread']

results = Parallel(n_jobs=num_cores, verbose=10)(delayed(retrain_and_save_agent)(
	agent_source=agent_source,
	hist=pd.DataFrame(hist[new_feature]),
	new_feature=new_feature,
	test_start_date='2017-05-01',
	bins=3,
	outputpath=os.path.join(folder, 'retrained')
	) for new_feature in features)
