from joblib import Parallel, delayed
import multiprocessing

from tqdm import tqdm, tqdm_notebook
import pandas as pd
import numpy as np
import math
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.append('..')
from helper.rl_framework import *
from helper.orderbook_container import OrderbookContainer
from helper.orderbook_trader import *
from helper.collect_samples import collect_samples_forward, collect_samples_backward
# from agents.NN_Agent import RLAgent_NN
from agents.BatchTree_Agent import RLAgent_BatchTree
from agents.QTable_Agent import QTable_Agent

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap

# Need to call function from same file, otherwise joblib.Parallel might run into an PicklingError
def unwrap_collect_samples_forward(**kwarg):
    return collect_samples_forward(**kwarg)

def unwrap_collect_samples_backward(**kwarg):
    return collect_samples_backward(**kwarg)

def trainer_QTable(orderbooks, V, T, period_length, vol_intervals, actions, limit_base,
                   agent_name='QTable_Agent', consume='cash', lim_stepsize=0.1,
                   state_variables=['volume', 'time'], normalized=False, interpolate_vol=False,
                   mode='backward'):
    brain = QTable_Agent(
        actions=actions,
        lim_stepsize=lim_stepsize,
        state_variables=state_variables,
        V=V, T=T, consume=consume,
        period_length=period_length,
        vol_intervals=vol_intervals,
        normalized=normalized,
        interpolate_vol=interpolate_vol,
        limit_base=limit_base,
        agent_name=agent_name
    )
    print(brain)

    brain.collect_samples_parallel(
        orderbooks=orderbooks, 
        mode=mode)

    print("learn from {} samples".format(len(brain.samples)))
    brain.learn_fromSamples()

    print("brain.samples.shape", brain.samples.shape)
    return brain

#@timing
def trainer_BatchTree(orderbooks, V, T, period_length, actions, limit_base, epochs,
            agent_name='BatchTree_Agent', consume='cash', lim_stepsize=0.1, 
            state_variables=['volume', 'time'], guiding_agent=None, random_start=True,
            mode='forward', retraining=24):
    brain = RLAgent_BatchTree(
        actions=actions,
        state_variables=state_variables,
        V=V, T=T, consume=consume,
        period_length=period_length,
        lim_stepsize=lim_stepsize,
        limit_base=limit_base,
        agent_name=agent_name
    )
    print(brain)

    print("Number of orderbook windows: {}".format(len(orderbooks)))

    splits = math.ceil(len(orderbooks)/retraining)
    for s in tqdm(range(splits)):
        print("Splitpoint: {}/{}".format(s, splits))
        start = s * retraining
        end = (s+1)*retraining
        orderbooks_sub = orderbooks[start:end]

        brain.collect_samples_parallel(
            orderbooks=orderbooks_sub, 
            mode=mode,
            epochs=epochs, 
            guiding_agent=guiding_agent,
            random_start=random_start,
            exploration=2)

        print("brain.learn_fromSamples() - {} samples".format(len(brain.samples)))
        brain.learn_fromSamples(nb_it=T*2, verbose=False, n_estimators=400, max_depth=20)

        print("brain.samples.shape", brain.samples.shape)
    return brain

