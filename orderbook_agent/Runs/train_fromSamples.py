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


def train_BatchTree_fromSamples(df, V, T, period_length, vol_intervals, actions,
                                state_variables, normalized=False, interpolate_vol=False,
                                n_estimators=20, max_depth=12, limit_base='incStepUnits'):
    brain = RLAgent_BatchTree(
        actions=actions,
        state_variables=state_variables,
        V=V, T=T,
        period_length=period_length,
        normalized=False,
        lim_stepsize=0.1,
        limit_base=limit_base
    )
    print(brain)
    
    brain.samples = df.copy()
    
    brain.fitted_Q_iteration_tree(nb_it=4, n_estimators=20, max_depth=12)
    
    return brain

def train_Qtable_fromSamples(samples, V, T, period_length, vol_intervals, actions,
                             state_variables=['volume', 'time'],
                             normalized=False, interpolate_vol=False):
    brain = QTable_Agent(
        actions=actions,
        state_variables=state_variables,
        V=V, T=T,
        period_length=period_length,
        vol_intervals=vol_intervals,
        normalized=normalized,
        interpolate_vol=interpolate_vol
    )
    print(brain)

    for tt in tqdm(range(T)[::-1]):
        #timepoint = period_length*tt
        #timepoint_next = min((tt+1)*period_length, (period_length*T)-1)
        time_left = T-tt
        
        df_sub = samples[samples.time==time_left]
        for idx, row in enumerate(df_sub.iterrows()):
 
            sample = row[1]

            state = brain.generate_state(time_left=sample.time, 
                                         volume_left=sample.volume)
            
            # new_state = brain.generate_state(
            #     time_left=sample.time_n,
            #     volume_left=sample.volume_n
            # )
            
            # rounded volume state
            new_state = brain.generate_state(
                time_left=sample.time_n, 
                volume_left=round_custombase(sample.volume_n, base=brain.volumes_base)
            )
            volume_traded_rounded = round_custombase(sample.volume-sample.volume_n, base=brain.volumes_base)
            cost = volume_traded_rounded * (sample.avg - sample.initial_center) / sample.initial_center

            # print("{}   {:1.2f}, {:1.4f}   {}".format(state, sample.action, cost, new_state))
            brain.learn(state=state, action=sample.action, cost=cost, new_state=new_state)
            
    return brain
