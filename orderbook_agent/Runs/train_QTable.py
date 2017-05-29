
import matplotlib as mpl

from tqdm import tqdm, tqdm_notebook
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
from agents.RL_Agent_Base import RLAgent_Base
from agents.QTable_Agent import QTable_Agent
from datetime import datetime
import fire
from IPython.display import display

def optimal_strategy_fixedStepsize(traingdata, V, T, period_length, vol_intervals, actions,
                     lim_stepsize, consume='volume',
                     verbose=True, state_variables=['volume', 'time'], limit_base='curr_ask',
                     outfile_agent=None, outfile_samples=None,
                     normalized=False, interpolate_vol=False):

    brain = QTable_Agent(
        actions=actions,
        lim_stepsize=lim_stepsize,
        state_variables=state_variables,
        V=V, T=T, consume=consume,
        period_length=period_length,
        vol_intervals=vol_intervals,
        normalized=normalized,
        interpolate_vol=interpolate_vol,
        limit_base=limit_base
    )
    print(brain)
    initial_cash = V
    limits = []
    for tt in tqdm_notebook(range(T)[::-1], desc='timepoints'):
        timepoint = period_length*tt
        timepoint_next = min((tt+1)*period_length, (period_length*T)-1)
        time_left = T-tt
        print("time_left", time_left)

        for w, window in tqdm_notebook(enumerate(traingdata), desc='tradingperiods', leave=False):
            ots = OrderbookTradingSimulator(orderbooks=window,
                                            volume=V, consume=consume, tradingperiods=T,
                                            period_length=period_length)

            initial_center = window[0].get_center()
            lim_increments = initial_center * (brain.lim_stepsize / 100)

            if consume=='volume':
                initial_marketprice, initial_limitWorst = window[0].get_current_price(volume=brain.V)
                initial_limitAvg = initial_marketprice / brain.V
                market_slippage = initial_limitAvg - initial_center
            elif consume=='cash':
                initial_marketShares = window[0].get_current_sharecount(cash=brain.V)
                initial_limitAvg = brain.V / initial_marketShares
            #print("V", brain.V, initial_limitAvg)
            
            ob_now = window[timepoint]
            ob_next = window[timepoint_next]
            timestamp = ob_now.timestamp

            price_incScale = int(ob_now.get_center()/lim_increments)

            ask = ob_now.get_ask()
            
            for vol in tqdm_notebook(brain.volumes, desc='volumes', leave=False):
                # # exceptionally do also sample volumes other than 100% at t=0,
                # # since we want to collect as many samples as possible
                if tt == 0 and vol != V:
                    # at t=0 we always have 100% of the volume left.
                    break
                
                state = brain.generate_state(time_left=time_left, 
                                             volume_left=vol,
                                             orderbook=ob_now)
                
                for action_idx, a in tqdm_notebook(enumerate(actions), desc='actions', leave=False):
                    ots.reset(custom_starttime=tt, custom_startvolume=vol)
                    # limit = ask * (1. + (a/100.))
                    #limit = initial_center * (1. + (a/100.))
                    #limit = initial_center + (a*lim_increments)
                    limit = lim_increments * (price_incScale + a)
                    limits.append(limit)

                    if brain.limit_base == 'agression':

                        summary = ots.trade(agression_factor=a, extrainfo={'ACTION':a})
                    else:
                        summary = ots.trade(limit=limit, extrainfo={'ACTION':a})  # agression_factor=a)

                    if consume=='volume':
                        volume_left = ots.volume
                        volume_traded = ots.history.volume_traded.values[-1]
                    elif consume=='cash':
                        volume_left = ots.cash
                        volume_traded = ots.history.cash_traded.values[-1]
                    
                    cost = ots.history.cost[-1]
                    new_state = brain.generate_state(time_left=time_left-1,
                                                     volume_left=volume_left, #_rounded,
                                                     orderbook=ob_next)
                    
                    # discrete lookup table needs discretized/rounded states
                    volume_left_rounded = round_custombase(volume_left, base=brain.volumes_base)
                    volume_traded_rounded = round_custombase(volume_traded, base=brain.volumes_base)
                    assert volume_left + volume_traded_rounded - vol*V <= 1.e-8, "{} {} {} {}".format(volume_left, volume_traded_rounded, vol, V)
                    
                    avg = ots.history.avg[-1]

                    if consume=='volume':
                        cost_rounded =  volume_traded_rounded * ((avg - initial_center) / market_slippage)
                    elif consume=='cash':
                        percentage = volume_traded_rounded / volume_traded
                        
                        extra_shares = ots.history.extra_shares.values[-1] * percentage
                        
                        cost_rounded = -extra_shares * avg
                    #print(time_left, vol, a, limit)
                    #print("volume vs rounded", volume_traded, volume_traded_rounded)
                    #print("cost vs rounded  ", cost, cost_rounded)
                    new_state_rounded = brain.generate_state(time_left=time_left-1,
                                                             volume_left=volume_left_rounded,
                                                             orderbook=ob_next)

                    # print("{}   {:1.2f}, {:1.4f}   {}".format(state, a, cost, new_state))
                    # print("{}   {:1.2f}, {:1.4f}   {}".format(state, a, cost_rounded, new_state_rounded))
                    # print("")
                    brain.learn(state=state, action=a, cost=cost_rounded, new_state=new_state_rounded)
                    brain.append_samples(
                        state=state,
                        action=a,
                        action_idx=action_idx,
                        cost=cost,
                        timestamp=timestamp,
                        avg=avg,
                        initial_center=initial_center,
                        new_state=new_state
                    )
                    #display(ots.history)
                
            if w%5==0 or w==len(traingdata)-1:
                # save model to disk
                brain.save(outfile_agent=outfile_agent, outfile_samples=outfile_samples)
            
    return brain, limits


def optimal_strategy(traingdata, V, T, period_length, vol_intervals, actions,
                     verbose=True, state_variables=['volume', 'time'], limit_base='curr_ask',
                     outfile_agent=None, outfile_samples=None,
                     normalized=False, interpolate_vol=False):

    brain = QTable_Agent(
        actions=actions,
        state_variables=state_variables,
        V=V, T=T,
        period_length=period_length,
        vol_intervals=vol_intervals,
        normalized=normalized,
        interpolate_vol=interpolate_vol,
        limit_base=limit_base
    )
    print(brain)
    limits = []
    for tt in tqdm_notebook(range(T)[::-1]):
        timepoint = period_length*tt
        timepoint_next = min((tt+1)*period_length, (period_length*T)-1)
        time_left = T-tt

        for w, window in tqdm_notebook(enumerate(traingdata)):
            ots = OrderbookTradingSimulator(orderbooks=window,
                                            volume=V, tradingperiods=T,
                                            period_length=period_length)

            initial_center = window[0].get_center()
            initial_marketprice, initial_limitWorst = window[0].get_current_price(brain.V)
            initial_limitAvg = initial_marketprice / brain.V
            market_slippage = initial_limitAvg - initial_center
            #print("V", brain.V, initial_limitAvg)
            
            ob_now = window[timepoint]
            ob_next = window[timepoint_next]
            timestamp = ob_now.timestamp

            ask = ob_now.get_ask()
            
            for vol in brain.volumes:
                
                # # exceptionally do also sample volumes other than 100% at t=0,
                # # since we want to collect as many samples as possible
                # if tt == 0 and vol != V:
                #     # at t=0 we always have 100% of the volume left.
                #     break
                
                state = brain.generate_state(time_left=time_left, 
                                             volume_left=vol,
                                             orderbook=ob_now)
                
                for action_idx, a in enumerate(actions):
                    ots.reset(custom_starttime=tt, custom_startvolume=vol)
                    
                    limit = ask * (1. + (a/100.))
                    # limit = initial_center * (1. + (a/100.))
                    limits.append(limit)
                    if brain.limit_base == 'agression':

                        summary = ots.trade(agression_factor=a, extrainfo={'ACTION':a})
                    else:
                        summary = ots.trade(limit=limit, extrainfo={'ACTION':a})  # agression_factor=a)

                    volume_left = ots.volume
                    volume_traded = ots.history.volume_traded.values[-1]
                    cost = ots.history.cost[-1]
                    new_state = brain.generate_state(time_left=time_left-1,
                                                     volume_left=volume_left, #_rounded,
                                                     orderbook=ob_next)

                    # discrete lookup table needs discretized/rounded states
                    volume_left_rounded = round_custombase(volume_left, base=brain.volumes_base)
                    volume_traded_rounded = round_custombase(volume_traded, base=brain.volumes_base)
                    assert volume_left + volume_traded_rounded - vol*V <= 1.e-8, "{} {} {} {}".format(volume_left, volume_traded_rounded, vol, V)

                    avg = ots.history.avg[-1]
                    
                    cost_rounded =  volume_traded_rounded * ((avg - initial_center) / market_slippage)
                    
                    new_state_rounded = brain.generate_state(time_left=time_left-1,
                                                             volume_left=volume_left_rounded,
                                                             orderbook=ob_next)

                    # print("{}   {:1.2f}, {:1.4f}   {}".format(state, a, cost, new_state))
                    # print("{}   {:1.2f}, {:1.4f}   {}".format(state, a, cost_rounded, new_state_rounded))
                    # print("")
                    brain.learn(state=state, action=a, cost=cost_rounded, new_state=new_state_rounded)
                    brain.append_samples(
                        state=state,
                        action=a,
                        action_idx=action_idx,
                        cost=cost,
                        timestamp=timestamp,
                        avg=avg,
                        initial_center=initial_center,
                        new_state=new_state
                    )
                    #display(ots.history)

            if w%5==0 or w==len(traingdata)-1:
                # save model to disk
                brain.save(outfile_agent=outfile_agent, outfile_samples=outfile_samples)
            
    return brain, limits

def run_old(inputfile, volume, volume_intervals, decision_points, period_length,
        action_min=-0.4, action_max=1.0, action_count=15, folder='experiments',
        state_variables=['volume', 'time'], limit_base='curr_ask', outfile_agent=None, outfile_samples=None):

    actions = list(np.linspace(action_min, action_max, num=action_count))
    print("V={}, T={}, P={}".format(volume, decision_points, period_length))
    print("Actions: ", ", ".join(["{:1.2f}".format(a) for a in actions]))
    
    inputfile_extension = inputfile.split(".")[-1]
    if inputfile_extension == "dict":
        episodes_train = OrderbookEpisodesGenerator(filename=inputfile,
                                                    episode_length=decision_points*period_length)  # [:20]
    elif inputfile_extension == "p":
        # saves a lot of time!
        episodes_train = pickle.load( open( inputfile, "rb" ) )
    print("Length of episodes_train: {}".format(len(episodes_train)))
    
    ql = optimal_strategy(traingdata=episodes_train, V=volume, T=decision_points,
                          period_length=period_length, vol_intervals=volume_intervals,
                          actions=actions, state_variables=state_variables, 
                          limit_base=limit_base,
                          outfile_agent=outfile_agent, outfile_samples=outfile_samples)

def run(inputfile, volume, volume_intervals, decision_points, period_length,
        actions, lim_stepsize=0.1, folder='experiments',
        state_variables=['volume', 'time'], limit_base='curr_ask', outfile_agent=None, outfile_samples=None):

    
    
    inputfile_extension = inputfile.split(".")[-1]
    if inputfile_extension == "dict":
        episodes_train = OrderbookEpisodesGenerator(filename=inputfile,
                                                    episode_length=decision_points*period_length)  # [:20]
    elif inputfile_extension == "p":
        # saves a lot of time!
        episodes_train = pickle.load( open( inputfile, "rb" ) )
    print("Length of episodes_train: {}".format(len(episodes_train)))
    
    ql = optimal_strategy_fixedStepsize(traingdata=episodes_train, V=volume, T=decision_points,
                          period_length=period_length, vol_intervals=volume_intervals,
                          actions=actions, lim_stepsize=lim_stepsize,
                          state_variables=state_variables, limit_base=limit_base,
                          outfile_agent=outfile_agent, outfile_samples=outfile_samples)

    
def main2():
    ## Settings
    experiment_name='1611_USDTBTC_Qtable_100vol10_60T4'
    inputfile='/home/axel/data/obs_2016-11_USDT_BTC_range1.2.dict'
    folder='experiments'
    outputfile_agent=os.path.join(outputfolder, experiment_name, 'model', experiment_name)
    outputfile_agent='q.json'
    outputfile_samples='q.csv'
    volume=100
    volume_intervals=10
    decision_points=4
    period_length=15
    action_min=-0.4
    action_max=1.0
    action_count=15
    state_variables=['volume','time','spread']
    limit_base='curr_ask'  # 'curr_ask', 'init_center'
    
    run(inputfile=inputfile, volume=volume,
        volume_intervals=volume_intervals, decision_points=decision_points,
        period_length=period_length, action_min=action_min, action_max=action_max,
        action_count=action_count, folder=folder, state_variables=state_variables,
        limit_base=limit_base,
        outfile_model=outfile_model, outfile_samples=outfile_samples)

if __name__ == '__main__':
    fire.Fire(run)


