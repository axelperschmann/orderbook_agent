import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import datetime
from IPython.display import display
import os
from tqdm import tqdm


import sys
sys.path.append('..')
from helper.orderbook_container import OrderbookContainer
from helper.orderbook_trader import OrderbookTradingSimulator


class RLAgent_Base:

    def __init__(self, actions, V, T, period_length, agent_name,
                 state_variables=['volume', 'time'], normalized=True):
        self.actions = actions
        self.state_dim = len(state_variables)
        self.action_dim = len(actions)

        self.T = T
        self.V = V
        self.period_length = period_length
        self.state_variables = state_variables
        self.created = datetime.datetime.now()
        self.agent_name = agent_name
        self.normalized = normalized

    def __str__(self):
        return("RL-Type: {}".format(type(self)))

    def save(self, outfile):
        raise NotImplementedError

    def load(infile):
        raise NotImplementedError
    
    def generate_state(self, time_left, volume_left, orderbook=None):   
        assert isinstance(time_left, (int, float)), "Parameter 'time_left' must be of type 'int', given: '{}'".format(type(time_left))
        assert isinstance(volume_left, (int, float)), "Parameter 'volume_left' must be of type 'int' or 'float', given: '{}'".format(type(volume_left))
        assert isinstance(orderbook, OrderbookContainer ) or orderbook is None, "Parameter 'orderbook' [if provided] must be of type 'Orderbook', given: '{}'".format(type(orderbook))

        allowed_variable_set = ['volume', 'time', 'spread']
        assert set(self.state_variables).issubset(allowed_variable_set), "Parameter 'state_variables' must be a subset of {}".format(allowed_variable_set)

        state = []
        for var in self.state_variables:
            if var == 'volume':
                if self.normalized:
                    state.append(volume_left/self.V)
                else:
                    state.append(volume_left)
            elif var == 'time':
                if self.normalized:
                    state.append(time_left/self.T)
                else:
                    state.append(time_left)
            elif var == 'spread':
                if orderbook is None:
                    state.append(0)
                    continue
                
                spread = orderbook.get_ask() - orderbook.get_bid()

                if spread <= 1.:
                    spread_discrete = 0
                elif spread > 2.:
                    spread_discrete = 2
                else:
                    spread_discrete = 1
                state.append(spread_discrete)
            else:
                raise NotImplemented

        return np.array(state)

    def get_action_index(self, action):
        action_idx = None
        assert action in self.actions, "Action not found: {}".format(action)
        
        action_idx = self.actions.index(action)

        return action_idx

    def choose_action(self, state, exploration=0):
        qval = self.predict(state)

        if random.random() < exploration:
            # choose random action
            action = random.choice(self.actions)
        else:
            # choose best action from Q(s,a) values
            action = self.actions[np.argmin(qval)]
        return action

    def predict(self, state):
        ''' this is only a skeleton '''
        raise NotImplementedError

    def learn(self, state, action, cost, new_state):
        ''' this is only a skeleton '''
        raise NotImplementedError

    def evaluate(self, testdata, verbose=False):
        costs = pd.DataFrame([])

        for w, window in tqdm(enumerate(testdata)):
            index = window[0].timestamp
            
            ## Learned strategy
            ots = OrderbookTradingSimulator(orderbooks=window, volume=self.V, tradingperiods=self.T,
                                             period_length=self.period_length)
            for t in range(0, self.T):
                time_left = self.T - t
                timepoint = t*self.period_length
                
                ob_now = window[timepoint]

                volume = float(ots.volume)
                if hasattr(self, 'volumes_base'):
                    # discretize volume through rounding (needed for QLookupTable)
                    volume = self.round_custombase(volume, non_zero=True)
    
                state = self.generate_state(time_left=time_left, 
                                            volume_left=volume,
                                            orderbook=ob_now)
                action, action_idx = self.get_action(state, exploration=0)
                limit = ob_now.get_ask() * (1. + (action/100.))
               
                ots.trade(limit = limit, extrainfo={'ACTION':action})

                if ots.summary['done']:
                    break
            if verbose:
                display(ots.history)
            costs.loc[index, self.agent_name] = ots.history.cost.sum()
        return costs

    def sample_from_Q(self, vol_intervals):
        assert len(self.state_variables) == 2, "Not yet implemented for more than 2 variables in state"

        df = pd.DataFrame([], columns=self.state_variables)
        for t in range(1, self.T+1):
            for v in np.linspace(0, self.V, num=vol_intervals+1)[1:]:
                state = self.generate_state(time_left=t,
                                        volume_left=v)

                q = self.predict(state)
                action = self.actions[np.nanargmin(q)]

                if (t, v) in [(1,100), (2,20), (1,10), (4,20)]:
                    print("t{}, v{}  -  action: #{}={:1.1f}".format(t,v, np.nanargmin(q), action))
                    print(["{:1.4f}".format(val) for val in q])
                
                df_tmp = pd.DataFrame({'time': t,
                                       'state': str(state),
                                       'volume': v,
                                       'q': np.nanmin(q),
                                       'action': action}, index=["{:1.2f},{}".format(v, t)])
                df = pd.concat([df, df_tmp])
        return df


    
    def heatmap_Q(self, hue='Q', vol_intervals=10, epoch=None, outfile=None, outformat='pdf', show_traces=False):
        assert len(self.state_variables) == 2, "Not yet implemented for more than 2 variables in state"
        
        df = self.sample_from_Q(vol_intervals=vol_intervals)
        
        fig, axs = plt.subplots(ncols=2, figsize=(16,5))
        plt.suptitle="X"
        
        sns.heatmap(df.pivot('time', 'volume', 'action'), annot=True, fmt="1.2f",
                    ax=axs[0], vmin=-0.4, vmax=1.0)
        sns.heatmap(df.pivot('time', 'volume', 'q'), annot=True, fmt="1.2f", ax=axs[1])
        title = "Q function (T:{}, V:{})".format(self.T*self.period_length, self.V)
        if epoch is not None:
            title = "{}, epochs:{}".format(title, epoch+1)

        if show_traces:
            for s, sample in self.samples.iterrows():
                x1, y1 = [sample.volume/vol_intervals, sample.volume_n/vol_intervals], [self.T-sample.time, self.T-sample.time_n]
                
                col = (len(self.actions)*(sample.action_idx)/256.,0.0, 0)
                
                # print(rgb)
                axs[1].plot(x1, y1, marker = 'o', color=col, alpha=0.3)
            
        for ax in axs:
            ax.invert_xaxis()
            ax.set_ylabel("time remaining [periods]")
            ax.set_xlabel("trade volume remaining [%]")
            # ax.set_zlabel("trade volume remaining [%]")
        axs[0].set_title('Optimal action')
        axs[1].set_title('Optimal Q value')
        
        if outfile:
            if outfile[len(outformat):] != outformat:
                outfile = "{}.{}".format(outfile, outformat)
            plt.savefig(outfile, format=outformat)
        else:
            plt.show()
        plt.close()
