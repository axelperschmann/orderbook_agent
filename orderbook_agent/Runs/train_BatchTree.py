from joblib import Parallel, delayed
import multiprocessing

from tqdm import tqdm, tqdm_notebook
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.append('..')
from helper.rl_framework import *
from helper.orderbook_container import OrderbookContainer
from helper.orderbook_trader import *
# from agents.NN_Agent import RLAgent_NN
from agents.BatchTree_Agent import RLAgent_BatchTree

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

#@timing
def trainer(orderbooks, V, T, period_length, actions, limit_base, epochs, consume='volume',
        lim_stepsize=0.1, state_variables=['volume', 'time'], random_start=True):
    brain = RLAgent_BatchTree(
        actions=actions,
        state_variables=state_variables,
        V=V, T=T, consume=consume,
        period_length=period_length,
        lim_stepsize=lim_stepsize,
        limit_base=limit_base,
    )
    print(brain)

    num_cores = multiprocessing.cpu_count()
    print("Start parallel collection of samples in forward mode (num_cores={})".format(num_cores))
    
    results = Parallel(n_jobs=num_cores, verbose=10)(delayed(collect_samples_forward)(brain=brain, window=window, epochs=epochs) for window in orderbooks)
    new_samples = pd.concat(results, axis=0, ignore_index=True)

    brain.append_samples(new_samples=new_samples)

    print("brain.fitted_Q_iteration_tree()")
    brain.fitted_Q_iteration_tree(nb_it=T*2, verbose=False)

    print("brain.samples.shape", brain.samples.shape)
    return brain

def collect_samples_forward(brain, window, epochs, random_start=True):
    ots = OrderbookTradingSimulator(orderbooks=window, volume=brain.V, consume=brain.consume,
                                    tradingperiods=brain.T, period_length=brain.period_length)
    lim_increments = ots.initial_center * (brain.lim_stepsize / 100)

    if brain.consume=='volume':
        initial_marketprice, initial_limitWorst = window[0].get_current_price(volume=brain.V)
    elif brain.consume=='cash':
        initial_marketShares, initial_limitWorst = window[0].get_current_sharecount(cash=brain.V)
    
    # empty DataFrame to store samples
    samples = pd.DataFrame([], columns=brain.columns)
    samples['action_idx'] = samples.action_idx.astype(int)

    for e in tqdm(range(epochs)):
        exploration = max(1.0/2**(e/20), 0.6)
        # print("{}: exploration = {}".format(e, exploration))

        volume = brain.V
        startpoint = 0
        if random_start and random.random() < 1.:
            # randomly start at other states in environment
            # volume = random.randint(1, brain.V)
            startpoint = random.randint(0, brain.T-1)
        ots.reset(custom_starttime=startpoint, custom_startvolume=volume)
        
        for t in range(startpoint, brain.T):
            time_left = brain.T - t

            ob_now = window[ots.t]
            ob_next = window[min(ots.t+brain.period_length, len(window)-1)]

            state = brain.generate_state(time_left=time_left, 
                                         volume_left=ots.get_units_left(),
                                         orderbook=ob_now)
            
            action, action_idx = brain.get_action(state, exploration)

            if brain.limit_base == 'currAsk':
                limit = ob_now.get_ask() * (1. + (action/100.))
                summary = ots.trade(limit=limit, verbose=False, extrainfo={'ACTION':action})
            elif brain.limit_base == 'incStepUnits':
                price_incScale = int(ob_now.get_center()/lim_increments)
                limit = lim_increments * (price_incScale + action)
                summary = ots.trade(limit=limit, verbose=False, extrainfo={'ACTION':action})
            elif brain.limit_base =='agression':
                summary = ots.trade(agression_factor=action, verbose=False, extrainfo={'ACTION':action})
            else:
                raise NotImplementedError

            new_state = brain.generate_state(time_left=time_left-1,
                                             volume_left=ots.get_units_left(),
                                             orderbook=ob_next)

            cost = ots.history.cost.values[-1]
            
            new_sample = brain.generate_sample(
                    state=state,
                    action=action,
                    action_idx=action_idx,
                    cost=cost,
                    timestamp=ob_now.timestamp,
                    avg=ots.history.avg[-1],
                    initial_center=ots.initial_center,
                    new_state=new_state
                )
            samples = pd.concat([samples, new_sample], axis=0, ignore_index=True)
            
            if ots.check_is_done():
                break
    return samples
