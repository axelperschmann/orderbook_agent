from tqdm import tqdm
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.append('..')
from helper.rl_framework import *
from helper.orderbook_container import OrderbookContainer
from helper.orderbook_trader import *
from agents.NN_Agent import RLAgent_NN
from agents.BatchTree_Agent import RLAgent_BatchTree


def trainer(orderbooks, V, T, period_length, actions, epochs, random_start=True, state_variables=['volume', 'time']):

    brain = RLAgent_BatchTree(
        actions=actions,
        state_variables=state_variables,
        V=V, T=T,
        period_length=period_length,
        normalized=False
    )
    print(brain)
    for i_window, window in (enumerate(orderbooks)):
    #window = orderbooks[0]
        ots = OrderbookTradingSimulator(orderbooks=window, volume=V,
                                        tradingperiods=T, period_length=period_length)
        for e in tqdm(range(epochs)):
            exploration = max(1.0/2**(e/20), 0.8)
            # print("{}: exploration = {}".format(e, exploration))

            volume = V
            startpoint = 0
            if random_start and random.random() < 1.:
                # randomly start at other states in environment
                volume = random.randint(1, V)
                startpoint = random.randint(0, T-1)
            
            ots.reset(custom_starttime=startpoint, custom_startvolume=volume)

            for t in range(startpoint, T)[:1]:

                time_left = T - t
                timepoint = t*period_length
                timepoint_next = min((t+1)*period_length, len(window)-1)
                ob_now = window[timepoint]
                ob_next = window[timepoint_next]
                        
                state = brain.generate_state(time_left=time_left, 
                                             volume_left=volume,
                                             orderbook=ob_now)
                
                # action, action_idx = brain.get_action(state, exploration)
                for action_idx, action in enumerate(actions):
                    ots.reset(custom_starttime=startpoint, custom_startvolume=volume)

                    limit = ob_now.get_ask() * (1. + (action/100.))
                    summary = ots.trade(limit=limit, verbose=False, extrainfo={'ACTION':action})
                    
                    new_state = brain.generate_state(time_left=time_left-1,
                                                     volume_left=float(ots.volume),
                                                     orderbook=ob_next)
                    cost = ots.history.cost.values[-1]

                    # print("{} {:1.2f} {:1.4f} {}".format(state, action, cost, new_state))

                    done = summary['done']

                    brain.append_samples(state, action, action_idx, cost, done, new_state)


        brain.fitted_Q_iteration_tree(nb_it=10)

    print("brain.samples.shape", brain.samples.shape)
    return brain


def train_RL(orderbooks, V, T, period_length, epochs, model=None, gamma=0.95, DECAY_RATE=0.005, epsilon=1.,
             bufferSize=50, batchSize=10, verbose=False, state_variables=['volume', 'time']):
    
    tmp = None
    post_observation = None
    k = 0
    actions = list(np.linspace(-0.4, 1.0, num=5))
    brain = RLAgent_NN(actions=actions, state_variables=state_variables, V=V, T=T, period_length=period_length)
    
    for i_window, window in tqdm(enumerate(orderbooks)):
        ots = OrderbookTradingSimulator(orderbooks=window, volume=V, tradingperiods=T,
                                                        period_length=period_length)
        
        for e in tqdm(range(epochs)):
            print("e", e)
            volume = V
            startpoint = 0
            
            if random.random() < 0.5:
                # randomly start at other states in environment
                volume = random.randint(1, V)
                startpoint = random.randint(0, T-1)
            ots.reset(custom_starttime=startpoint, custom_startvolume=volume)
            exploration = max(1.0/2**(e/2), 0)
            print("exploration {} = {:1.2f}".format(e, exploration))
            
            for step in range(startpoint, T):
                time_left = T - step
                timepoint = step*period_length
                timepoint_next = min((step+1)*period_length, len(window)-1)
                
                ob_now = window[timepoint]
                ob_next = window[timepoint_next]
                
                if post_observation is not None:
                    pre_observation = post_observation
                if clf is None:
                    action = random.choice(actions)
                else:
                    action = get_action(clf, pre_observation, exploration, env.action_space)
                print("action: ",action)
                state = brain.generate_state(time_left=time_left,
                                         volume_left=volume,
                                         orderbook=ob_now)

                limit = ob_now.get_ask() * (1. + (action/100.))
                summary = ots.trade(limit=limit, verbose=False, extrainfo={'ACTION':action})  #agression_factor=action

                volume = float(ots.volume)
                new_state = brain.generate_state(time_left=time_left-1,
                                                 volume_left=volume,
                                                 orderbook=ob_next)
                cost = ots.history.cost.values[-1]

                print("point", get_point(state, action, cost, new_state))
                
            #print(info)
    return brain