import matplotlib as mpl
mpl.use('Agg')

from tqdm import tqdm
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline

import sys
sys.path.append('..')
from helper.orderbook_trader import OrderbookTradingSimulator
from helper.Q_learning import QLearn, round_custombase
from helper.manage_orderbooks_v2 import OrderbookEpisodesGenerator
from datetime import datetime


ql = QLearn.load("evaluation/models/1611-1701_USDTBTC_Qtable_100vol10_60T4.json")
ql.plot_Q()

T = ql.T
V = ql.V
period_length = ql.period_length
actions = ql.actions

# filename_train = '/home/axel/data/obs_2017-01_USDT_BTC_range1.2.dict'
# episodes_train = OrderbookEpisodesGenerator(filename=filename_train, episode_length=T*period_length)
# print("Length of episodes_train: {}".format(len(episodes_train)))

outfile = "evaluation/Evaluation_obs_TESTSET_2017-02_USDT_BTC.csv"
filename_test = '/home/axel/data/obs_2017-02_USDT_BTC_range1.2.dict'
episodes_test = OrderbookEpisodesGenerator(filename=filename_test, episode_length=T*period_length)
print("Length of episodes_test: {}".format(len(episodes_test)))



def run_Q(V, H, T, ql, episode_windows, actions):
    costs = pd.DataFrame([])
    period_length = int(H/T)

    for e, episode in tqdm(enumerate(episode_windows)):
        index = episode[0].timestamp
        
        volume = V
        
        ## Learned strategy
        ots = OrderbookTradingSimulator(orderbooks=episode, volume=volume, tradingperiods=T,
                                         period_length=period_length)
        for tt in range(1, T+1, 1)[::-1]:
            new_vol = round_custombase(ots.volume, base=ql.vol_intervals)  
        
            if new_vol > 0:
                state = ql.state_as_string(time_left=tt, volume_left=new_vol/V)
        
                action = ql.chooseAction(state)
                if tt == 4:
                    action = 0.1
        
        
                # print(state, action)
                obs = episode[period_length * (T-tt)].copy()
                # obs = [elem.copy() for elem in obs_]
            
                ask = obs.get_ask()
                # center = ots.masterbook.get_center()
                limit = ask * (1. + (action/100.))
            else:
                # theoreticall done
                limit == None
            ots.trade(limit = limit, extrainfo={'ACTION':action})
        costs.loc[index, 'Learned'] = ots.history.cost.sum()
        
        # for a in actions:
        a = 0.4
        lim = episode[0].get_ask() * (1. + (a/100.))
        # print("\n### Fixed limit at: {} (ASK+4) ###".format(lim))
        ots = OrderbookTradingSimulator(orderbooks=episode, volume=volume, tradingperiods=1,
                                        period_length=period_length*T)
        ots.trade(limit = lim)
        key = 'ask*{:1.3f}'.format((1. + (a/100.)))
        costs.loc[index, key] = ots.history.cost.sum()
        
        ## market order
        ots = OrderbookTradingSimulator(orderbooks=episode, volume=volume, tradingperiods=T,
                                        period_length=period_length)
        ots.trade(limit = None)
        costs.loc[index, 'Market'] = ots.history.cost.sum()
        
    return costs

print("Evaluate Testset")
costs_list_train = run_Q(V=100, H=T*period_length, T=T, ql=ql, episode_windows=episodes_test, actions=actions)
costs_list_train.to_csv(outfile)
