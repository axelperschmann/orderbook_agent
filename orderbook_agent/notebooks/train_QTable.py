
import matplotlib as mpl

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
from agents.QTable_Agent import QTable_Agent
from datetime import datetime
import fire
from IPython.display import display

def optimal_strategy(traingdata, V, T, period_length, vol_intervals, actions, outputfolder, verbose=True, experiment_name=None, state_variables=['volume', 'time'], plotQ=False, outputfile_model=None):

    brain = QTable_Agent(
        actions=actions,
        state_variables=state_variables,
        V=V, T=T,
        period_length=period_length,
        vol_intervals=vol_intervals,
        normalized=True
    )
    print(brain)
    
    if outputfile_model is not None:
        filename_qtable = outputfile_model
    else:
        filename_qtable = os.path.join(outputfolder, experiment_name, 'model', experiment_name)
    filename_graphs = os.path.join(outputfolder, experiment_name, 'graphs', experiment_name)

    for tt in tqdm(range(T)[::-1]):
        
        timepoint = period_length*tt
        timepoint_next = min((tt+1)*period_length, (period_length*T)-1)
        time_left = T-tt

        for w, window in tqdm(enumerate(traingdata)):
            ots = OrderbookTradingSimulator(orderbooks=window,
                                            volume=V, tradingperiods=T,
                                            period_length=period_length)

            initial_center = window[0].get_center()
            
            ob_now = window[timepoint]
            ob_next = window[timepoint_next]

            ask = ob_now.get_ask()
            
            for vol in brain.volumes:
                
                if tt == 0 and vol != V:
                    # at t=0 we always have 100% of the volume left.
                    break

                state = brain.generate_state(time_left=time_left, 
                                             volume_left=vol,
                                             orderbook=ob_now)
                
                for a in actions:
                    ots.reset(custom_starttime=tt, custom_startvolume=vol)
                    
                    limit = ask * (1. + (a/100.))

                    ots.trade(limit = limit)  # agression_factor=a)
                    
                    volume_left = ots.volume
                    volume_left_rounded = round_custombase(volume_left, base=brain.volumes_base)

                    volume_traded = ots.history.volume_traded.values[-1]
                    volume_traded_rounded = round_custombase(volume_traded, base=brain.volumes_base)

                    assert volume_left_rounded + volume_traded_rounded - vol*V <= 1.e-8, "{} {} {} {}".format(volume_left_rounded, volume_traded_rounded, vol, V)

                    avg = ots.history.avg[-1]

                    # manually compute costs, since we have to think in discrete volume steps (rounding ...)
                    cost = volume_traded_rounded * (avg - initial_center) / initial_center
                    
                    new_state = brain.generate_state(time_left=time_left-1,
                                                     volume_left=volume_left_rounded,
                                                     orderbook=ob_next)

                    # print("{}   {:1.2f}, {:1.4f}   {}".format(state, a, cost, new_state))
                    
                    brain.learn(state=state, action=a, cost=cost, new_state=new_state)

            if w%5==0 or w==len(traingdata):
                if plotQ:
                    # brain.plot_Q(outfile="{}_{}_action".format(filename_graphs, T-tt), epoch=w,
                    #           z_represents='action', verbose=verbose)
                    # brain.plot_Q(outfile="{}_{}_Q".format(filename_graphs, T-tt), epoch=w, 
                    #           z_represents='Q', verbose=verbose)
                    pass

                brain.save(outfile=filename_qtable)
            
    return brain

def run(experiment_name, inputfile, volume, volume_intervals, decision_points, periodlength,
        action_min=-0.4, action_max=1.0, action_count=15, folder='experiments', plotQ=False, state_variables=['volume', 'time']):
    
    actions = list(np.linspace(action_min, action_max, num=action_count))
    print("V={}, T={}, P={}".format(volume, decision_points, periodlength))
    print("Actions: ", ", ".join(["{:1.2f}".format(a) for a in actions]))
    
    episodes_train = OrderbookEpisodesGenerator(filename=inputfile,
                                                episode_length=decision_points*periodlength)
    print("Length of episodes_train: {}".format(len(episodes_train)))
    
    ql = optimal_strategy(traingdata=episodes_train, V=volume, T=decision_points,
                          period_length=periodlength, vol_intervals=volume_intervals,
                          actions=actions, outputfolder=folder, experiment_name=experiment_name,
                          state_variables=state_variables, plotQ=plotQ)

    
def main2():
    ## Settings
    experiment_name='1611_USDTBTC_Qtable_100vol10_60T4'
    inputfile='/home/axel/data/obs_2016-11_USDT_BTC_range1.2.dict'
    volume=100
    volume_intervals=10
    decision_points=4
    period_length=15
    action_min=-0.4
    action_max=1.0
    action_count=2
    folder='experiments'
    plotQ=True
    state_variables=['volume','time','spread']
    
    run(experiment_name=experiment_name, inputfile=inputfile, volume=volume,
        volume_intervals=volume_intervals, decision_points=decision_points,
        period_length=period_length, action_min=action_min, action_max=action_max,
        action_count=action_count, folder=folder, plotQ=plotQ, state_variables=state_variables)

if __name__ == '__main__':
    fire.Fire(run)


