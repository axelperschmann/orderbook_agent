import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm import tqdm
from sklearn import preprocessing

#sys.path.append('../Runs')
sys.path.append('..')
from agents.QTable_Agent import QTable_Agent
from agents.BatchTree_Agent import RLAgent_BatchTree
from helper.Q_learning import round_custombase


def train_BatchTree_fromSamples(df, V, T, consume, period_length, actions, limit_base, 
                                state_variables, n_estimators=20, max_depth=12):
    brain = RLAgent_BatchTree(
        actions=actions,
        state_variables=state_variables,
        V=V, T=T, consume=consume,
        period_length=period_length,
        lim_stepsize=0.1,
        limit_base=limit_base
    )
    print(brain)
    
    brain.samples = df.copy()
    
    brain.fitted_Q_iteration_tree(nb_it=T*2, n_estimators=20, max_depth=12)
    
    return brain

def train_Qtable_fromSamples(samples, V, T, consume, period_length, actions, limit_base,
                             vol_intervals, state_variables=['volume', 'time'],
                             normalized=True, interpolate_vol=False):
    brain = QTable_Agent(
        actions=actions,
        state_variables=state_variables,
        V=V, T=T, consume=consume,
        period_length=period_length,
        vol_intervals=vol_intervals,
        normalized=normalized,
        interpolate_vol=interpolate_vol,
        lim_stepsize=0.1,
        limit_base=limit_base
    )
    print(brain)

    for tt in tqdm(range(T)[::-1]):
        #timepoint = period_length*tt
        #timepoint_next = min((tt+1)*period_length, (period_length*T)-1)
        time_left = T-tt
        
        df_sub = samples[samples.time==time_left]
        display(df_sub.head())
        for idx, row in enumerate(df_sub.iterrows()):
            sample = row[1]


            state = sample[state_variables].copy()
            state['volume'] = round_custombase(state.volume, base=brain.volumes_base)
            if normalized:
                state['volume'] = state['volume'] / V
                state['time'] = state['time'] / T
            state = list(state.values)
            
            new_state = sample[["{}_n".format(var) for var in state_variables]]
            new_state['volume_n'] = round_custombase(new_state.volume_n, base=brain.volumes_base)
            if normalized:
                new_state['volume_n'] = new_state['volume_n'] / V
                new_state['time_n'] = new_state['time_n'] / T
            new_state = list(new_state.values)

            volume_traded = sample.volume-sample.volume_n
            volume_traded_rounded = round_custombase(volume_traded, base=brain.volumes_base)
            if volume_traded == 0:
                cost = 0
            else:
                cost = sample.cost / volume_traded * volume_traded_rounded

            # print("{}   {:1.2f}, {:1.4f}   {}".format(state, sample.action, cost, new_state))
            brain.learn(state=state, action=sample.action, cost=cost, new_state=new_state)
            
    return brain

