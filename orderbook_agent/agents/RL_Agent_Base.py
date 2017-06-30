from joblib import Parallel, delayed
import multiprocessing

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import datetime
from IPython.display import display, clear_output
import os
import json
from tqdm import tqdm, tqdm_notebook
import dill, pickle
from helper.collect_samples import collect_samples_forward, collect_samples_backward


import sys
sys.path.append('..')
from helper.orderbook_container import OrderbookContainer
from helper.orderbook_trader import OrderbookTradingSimulator
from helper.general_helpers import safe_list_get


def unwrap_collect_samples_forward(**kwarg):
    # kwarg['brain'] = dill.dumps(kwarg['brain'])
    return collect_samples_forward(**kwarg)

def unwrap_collect_samples_backward(**kwarg):
    return collect_samples_backward(**kwarg)


class RLAgent_Base:

    def __init__(self, actions, lim_stepsize, V, T, consume, period_length, samples, 
                 agent_name, agent_type, limit_base, state_variables=['volume', 'time'],
                 normalized=False):
        self.actions = list(actions)
        self.state_dim = len(state_variables)
        self.action_dim = len(actions)
        self.lim_stepsize = lim_stepsize or 0.1

        self.T = T
        self.V = V
        self.consume = consume
        self.period_length = period_length
        self.state_variables = state_variables
        self.created = datetime.datetime.now()
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.normalized = normalized
        self.limit_base = limit_base

        self.columns = state_variables + ['action', 'action_idx', 'cost', 'avg', 'initial_center', 'timestamp'] + [var + "_n" for var in state_variables]
        self.samples = pd.DataFrame(samples, columns=self.columns)
        self.samples['action_idx'] = self.samples.action_idx.astype(int)

    def __str__(self):
        return("RL-Type: {}, Name: '{}', state_variables: '{}'".format(type(self), self.agent_name, list(self.state_variables)))

    def generate_sample(self, state, action, action_idx, cost, avg, initial_center, timestamp, new_state):
        return pd.DataFrame([state + [round(action, 2)] + [action_idx] + [cost] + [avg] + [initial_center] + [pd.to_datetime(timestamp)] + new_state],
            columns=self.columns)

    def append_samples(self, new_samples):
        self.samples = pd.concat([self.samples, new_samples], axis=0, ignore_index=True)

    def collect_samples_parallel(self, orderbooks, epochs=20, random_start=False, exploration=0, mode='forward', append=True, limit_num_cores=None):
        assert isinstance(mode, str) and mode in ['forward', 'backward'], "Unknown sample collection mode: '{}'".format(mode)
        if limit_num_cores:
            num_cores = limit_num_cores
        else:
            num_cores = multiprocessing.cpu_count()
        
        print("Start parallel collection of samples in '{}' mode (num_cores={})".format(mode, num_cores))

        if mode=='forward':
            results = Parallel(n_jobs=num_cores, verbose=10)(delayed(unwrap_collect_samples_forward, check_pickle=False)(
                                                                brain=self, window=window, epochs=epochs, random_start=random_start, exploration=exploration) 
                                                                    for window in orderbooks)
        elif mode=='backward':
            results = Parallel(n_jobs=num_cores, verbose=10)(delayed(unwrap_collect_samples_backward)(
                                                                brain=self, window=window) 
                                                                    for window in orderbooks)

        new_samples = pd.concat(results, axis=0, ignore_index=True).drop_duplicates()
        self.append_samples(new_samples=new_samples)

        return new_samples

    def copy(self):
        raise NotImplementedError

    def save(self, outfile, outfile_samples):
        raise NotImplementedError

    def load(agent_name=None, path='.', infile_agent=None, infile_model=None, infile_samples=None, ignore_samples=False):
        if agent_name is None:
            assert isinstance(infile_agent, str), "Bad parameter 'infile_agent', given: {}".format(infile_agent)
        else:
            infile_agent = "{}.json".format(agent_name)
            infile_model = "{}.obj".format(agent_name)
            infile_samples = "{}.csv".format(agent_name)

        with open(os.path.join(path, infile_agent), 'r') as f:
            data = json.load(f)

        agent_type = safe_list_get(data, idx='agent_type', default='QTable_Agent')
        
        if agent_type=='QTable_Agent':
            from agents.QTable_Agent import QTable_Agent
            agent = QTable_Agent.load(
                agent_name=agent_name,
                path=path,
                infile_agent=infile_agent,
                infile_samples=infile_samples,
                ignore_samples=ignore_samples
                )
        elif agent_type=='BatchTree_Agent':
            from agents.BatchTree_Agent import RLAgent_BatchTree
            agent = RLAgent_BatchTree.load(
                agent_name=agent_name,
                path=path,
                infile_agent=infile_agent,
                infile_model=infile_model,
                infile_samples=infile_samples,
                ignore_samples=ignore_samples
                )
        elif agent_type=='NN_Agent':
            from agents.NN_Agent import RLAgent_NN
            agent = RLAgent_NN.load(
                agent_name=agent_name,
                path=path,
                infile_agent=infile_agent,
                infile_model=infile_model,
                infile_samples=infile_samples,
                ignore_samples=ignore_samples
                )
        else:
            raise ValueError("Unknown agent_type, given: '{}'".format(agent_type))

        return agent

    
    def generate_state(self, time_left, volume_left, orderbook=None, orderbook_cheat=None, extra_variables=None):  
        assert isinstance(time_left, (int, float)), "Parameter 'time_left' must be of type 'int', given: '{}'".format(type(time_left))
        assert isinstance(volume_left, (int, np.int64, float, np.float)), "Parameter 'volume_left' must be of type 'int' or 'float', given: '{}'".format(type(volume_left))
        assert (type(orderbook).__name__ == OrderbookContainer.__name__) or orderbook is None, "Parameter 'orderbook' [if provided] must be of type 'Orderbook', given: '{}'".format(type(orderbook))

        if extra_variables is not None:
            for key in extra_variables.keys():
                assert key in self.state_variables, "extra_variables '{}' is not contained in agents state_variables: {}".format(key, self.state_variables)
        # allowed_variable_set = ['volume', 'time', 'spread']
        # assert set(self.state_variables).issubset(allowed_variable_set), "Parameter 'state_variables' must be a subset of {}".format(allowed_variable_set)

        state = []
        for feat in self.state_variables:
            if (extra_variables is not None) and feat in extra_variables:
                # useful if we want to plot heatmap for agents with more than 2 variables.
                state.append(extra_variables[feat])
                continue

            if feat == 'volume':
                if self.normalized:
                    state.append(float(volume_left/self.V))
                else:
                    state.append(float(volume_left))
            elif feat == 'time':
                if self.normalized:
                    state.append(float(time_left/self.T))
                else:
                    state.append(float(time_left))
            elif feat == 'spread':
                state.append(float(orderbook.features['spread']))
            elif feat == 'shares':
                val = orderbook.get_current_sharecount(cash=self.V)[0]
                state.append(val)
            elif feat == 'future15_market':
                fut_sharecount = orderbook_cheat.get_current_sharecount(cash=self.V)
                sharecount = orderbook.get_current_sharecount(cash=self.V)
                val = (fut_sharecount[0]/sharecount[0]) - 1
                state.append(val)
            elif feat == 'future15_ob':
                fut_price = orderbook_cheat.get_center()
                price = orderbook.get_center()
                val = (fut_price/price) - 1
                state.append(val)

            else:
                val = float(orderbook.features[feat])

                state.append(val)
                
            #else:
            #    raise NotImplemented

        return list(state)

    def get_action_index(self, action):
        action_idx = None
        assert action in self.actions, "Action not found: {} {}".format(action, self.actions)
        
        action_idx = self.actions.index(action)

        return action_idx

    def action_to_limit(self, action, init_center=None, current_ask=None):
        assert (self.limit_base == 'init_center' and isinstance(init_center, float)) or \
                (self.limit_base == 'currAsk' and isinstance(current_ask, float)), "Given: {}".format(self.limit_base)

        if self.limit_base == 'init_center':
            limit = init_center * (1. + (action/100.))
        elif self.limit_base == 'currAsk':
            limit = current_ask * (1. + (action/100.))
        else:
            raise ValueError

        return limit


    # def choose_action(self, state, exploration=0):
    #     qval = self.predict(state)
    # 
    #     if random.random() < exploration:
    #         # choose random action
    #         action = random.choice(self.actions)
    #     else:
    #         # choose best action from Q(s,a) values
    #         action = self.actions[np.argmin(qval)]
    #     return action

    def predict(self, state):
        ''' this is only a skeleton '''
        raise NotImplementedError

    def learn(self, state, action, cost, new_state):
        ''' this is only a skeleton '''
        raise NotImplementedError

    def sample_from_Q(self, vol_intervals, which_min):
        ''' this is only a skeleton '''
        raise NotImplementedError
    
    def heatmap_Q(self, hue='Q', vol_intervals=10, epoch=None, which_min='first', outfile=None, outformat='pdf', show_traces=False, show_minima_count=False, extra_variables=None):
        # assert len(self.state_variables) == 2, "Not yet implemented for a statespace with more than 2 dimensions"
        
        if show_minima_count:
            fig, axs = plt.subplots(ncols=3, figsize=(16,4))
        else:
            fig, axs = plt.subplots(ncols=2, figsize=(16,5))

        df = self.sample_from_Q(vol_intervals=vol_intervals, which_min=which_min, extra_variables=extra_variables)
        if len(df) == 0:
            raise ValueError("Could not sample anything")

        sns.heatmap(df.pivot('time', 'volume', 'action'), annot=True, fmt="1.1f",
                    ax=axs[0], vmin=self.actions[0], vmax=self.actions[-1]) 
        axs[0].set_title('Optimal action')
        
        sns.heatmap(df.pivot('time', 'volume', 'q'), annot=True, annot_kws={'rotation':90}, fmt="1.2f", ax=axs[1])
        axs[1].set_title('Optimal Q value')
        
        if show_minima_count:
            sns.heatmap(df.pivot('time', 'volume', 'minima_count'), annot=True, ax=axs[2])
            axs[2].set_title('Minima Count')
        
        title = "{}: Q function (T:{}, V:{})".format(self.agent_name, self.T*self.period_length, self.V)
        if epoch is not None:
            title = "{}, epochs:{}".format(title, epoch+1)
        if extra_variables is not None:
            title = "{}, extra_variables: {}".format(title, extra_variables)
        fig.suptitle(title)

        if show_traces:
            first = True
            scale_factor = 1./self.V*vol_intervals
            for s, sample in self.samples.iterrows():
                x1, y1 = [sample.volume*scale_factor, sample.volume_n*scale_factor], [self.T-sample.time, self.T-sample.time_n]
                col = (len(self.actions)*(sample.action_idx)/256.,0.0, 0)
                axs[1].plot(x1, y1, marker = 'o', color=col, alpha=0.3)
            
        for ax in axs:
            ax.invert_xaxis()
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_ylabel("time remaining [periods]")
            ax.set_xlabel("trade volume remaining [%]")
            # ax.set_zlabel("trade volume remaining [%]")
            # ax.set_ylim((1,4))
        plt.tight_layout()
        if outfile:
            if outfile[len(outformat):] != outformat:
                outfile = "{}.{}".format(outfile, outformat)
            plt.savefig(outfile, format=outformat)
        else:
            plt.show()
        plt.close()
