
import matplotlib as mpl
mpl.use('Agg')

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
sys.path.append('..')
from helper.orderbook_trader import OrderbookTradingSimulator
from helper.Q_learning import QLearn, round_custombase
from helper.manage_orderbooks import OrderbookEpisodesGenerator
from datetime import datetime
import fire


def optimal_strategy(traingdata, V, T, decisionfrequency, vol_intervals, actions, outputfolder, verbose=True, ql=None, experiment_name=None, state_variables=['volume', 'time'], plotQ=False):
    timestamp = datetime.now()
    
    print("V: {}, T: {}, decisionfrequency: {}, vol_intervals: {}, num_actions: {}"
          .format(V, T, decisionfrequency, vol_intervals, len(actions)))
    print("actions: {}".format(actions))
    volumes = np.linspace(1, 0, num=vol_intervals+1)[:-1] # skip volumes=0
    
    volumes_base = float(V)/vol_intervals

    print("volumes: {}".format(volumes))
    
    ql = ql or QLearn(actions=actions, vol_intervals=vol_intervals, V=V, T=T,
                      decisionfrequency=decisionfrequency, state_variables=state_variables)
    
    filename_qtable = os.path.join(outputfolder, experiment_name, 'model', experiment_name)
    filename_graphs = os.path.join(outputfolder, experiment_name, 'graphs', experiment_name)

    for tt in tqdm(range(T)[::-1]):
        trading_startpoint = decisionfrequency*tt
        time_left = T-tt

        for e, episode in tqdm(enumerate(traingdata)):
            initial_center = episode[0].get_center()
            
            center = episode[trading_startpoint].get_center()
            ask = episode[trading_startpoint].get_ask()
            
            for vol in volumes:
                if tt == 0 and vol != 1.:
                    # at t=0 we always have 100% of the volume left.
                    break
                
                state = ql.state_as_string(time_left=time_left, volume_left=vol,
                                           orderbook=episode[trading_startpoint])

                for a in actions:

                    ots = OrderbookTradingSimulator(orderbooks=episode[trading_startpoint:],
                                                    volume=vol*V, tradingperiods=T-tt,
                                                    decisionfrequency=decisionfrequency)
                    limit = ask * (1. + (a/100.))

                    ots.trade(limit = limit)  # agression_factor=a)

                    volume_left = ots.volume
                    volume_left_rounded = round_custombase(volume_left, base=volumes_base)

                    volume_traded = ots.history.volume_traded.values[-1]
                    volume_traded_rounded = round_custombase(volume_traded, base=volumes_base)

                    assert volume_left_rounded + volume_traded_rounded - vol*V <= 1.e-8, "{} {} {} {}".format(volume_left_rounded, volume_traded_rounded, vol, V)

                    avg = ots.history.avg[-1]

                    # manually compute costs, since we have to think in discrete volume steps (rounding ...)
                    cost = volume_traded_rounded * (avg - initial_center) / initial_center
                    
                    next_ob = episode[trading_startpoint+decisionfrequency-1]  # or ots.masterbook?!
                    new_state = ql.state_as_string(time_left=time_left-1,
                                                volume_left=volume_left_rounded/V,
                                                orderbook=next_ob)
                    
                    ql.learn(state, a, cost, new_state)

            if e%5==0 or e==len(traingdata):
                if plotQ:
                    ql.plot_Q(outfile="{}_{}_action".format(filename_graphs, T-tt), epoch=e,
                              z_represents='action', verbose=verbose)
                    ql.plot_Q(outfile="{}_{}_Q".format(filename_graphs, T-tt), epoch=e, 
                              z_represents='Q', verbose=verbose)

                ql.save(outfile=filename_qtable)
            
    return ql

def run(experiment_name, inputfile, volume, volume_intervals, decision_points, periodlength,
        action_min=-0.4, action_max=1.0, action_count=15, folder='experiments', plotQ=False, state_variables=['volume', 'time']):
    
    actions = list(np.linspace(action_min, action_max, num=action_count))
    print("V={}, T={}, P={}".format(volume, decision_points, periodlength))
    print("Actions: ", ", ".join(["{:1.2f}".format(a) for a in actions]))
    
    episodes_train = OrderbookEpisodesGenerator(filename=inputfile,
                                                episode_length=decision_points*periodlength)
    print("Length of episodes_train: {}".format(len(episodes_train)))
    
    ql = optimal_strategy(traingdata=episodes_train, V=volume, T=decision_points,
                          decisionfrequency=periodlength, vol_intervals=volume_intervals,
                          actions=actions, outputfolder=folder, experiment_name=experiment_name,
                          state_variables=state_variables, plotQ=plotQ)

    
def main2():
    ## Settings
    experiment_name='1611_USDTBTC_Qtable_100vol10_60T4'
    inputfile='/home/axel/data/obs_2016-11_USDT_BTC_range1.2.dict'
    volume=100
    volume_intervals=10
    decision_points=4
    periodlength=15
    action_min=-0.4
    action_max=1.0
    action_count=2
    folder='experiments'
    plotQ=True
    state_variables=['volume','time','spread']
    
    run(experiment_name=experiment_name, inputfile=inputfile, volume=volume,
        volume_intervals=volume_intervals, decision_points=decision_points,
        periodlength=periodlength, action_min=action_min, action_max=action_max,
        action_count=action_count, folder=folder, plotQ=plotQ, state_variables=state_variables)

if __name__ == '__main__':
    fire.Fire(run)


