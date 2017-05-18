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
from tqdm import tqdm, tqdm_notebook


import sys
sys.path.append('..')
from helper.orderbook_container import OrderbookContainer
from helper.orderbook_trader import OrderbookTradingSimulator


class RLAgent_Base:

    def __init__(self, actions, lim_stepsize, V, T, period_length, samples, agent_name, limit_base,
                 state_variables=['volume', 'time'], normalized=True):
        self.actions = actions
        self.state_dim = len(state_variables)
        self.action_dim = len(actions)
        self.lim_stepsize = lim_stepsize or 0.1

        self.T = T
        self.V = V
        self.period_length = period_length
        self.state_variables = state_variables
        self.created = datetime.datetime.now()
        self.agent_name = agent_name
        self.normalized = normalized
        self.limit_base = limit_base

        self.columns = state_variables + ['action', 'action_idx', 'cost', 'avg', 'initial_center', 'timestamp'] + [var + "_n" for var in state_variables]
        self.samples = pd.DataFrame(samples, columns=self.columns)
        self.samples['action_idx'] = self.samples.action_idx.astype(int)

    def __str__(self):
        return("RL-Type: {}".format(type(self)))

    def append_samples(self, state, action, action_idx, cost, avg, initial_center, timestamp, new_state):
        tmp = pd.DataFrame([state.tolist() + [round(action, 2)] + [action_idx] + [cost] + [avg] + [initial_center] + [pd.to_datetime(timestamp)] + new_state.tolist()], columns=self.columns)
        self.samples = pd.concat([self.samples, tmp], axis=0, ignore_index=True)

    def save(self, outfile, outfile_samples):
        raise NotImplementedError

    def load(infile_agent, infile_samples):
        raise NotImplementedError
    
    def generate_state(self, time_left, volume_left, orderbook=None):  
        assert isinstance(time_left, (int, float)), "Parameter 'time_left' must be of type 'int', given: '{}'".format(type(time_left))
        assert isinstance(volume_left, (int, np.int64, float, np.float)), "Parameter 'volume_left' must be of type 'int' or 'float', given: '{}'".format(type(volume_left))
        assert (type(orderbook).__name__ == OrderbookContainer.__name__) or orderbook is None, "Parameter 'orderbook' [if provided] must be of type 'Orderbook', given: '{}'".format(type(orderbook))

        # allowed_variable_set = ['volume', 'time', 'spread']
        # assert set(self.state_variables).issubset(allowed_variable_set), "Parameter 'state_variables' must be a subset of {}".format(allowed_variable_set)

        state = []
        for feat in self.state_variables:
            if feat == 'volume':
                if self.normalized:
                    state.append(volume_left/self.V)
                else:
                    state.append(volume_left)
            elif feat == 'time':
                if self.normalized:
                    state.append(time_left/self.T)
                else:
                    state.append(time_left)
            elif feat == 'spread':
                state.append(orderbook.features['spread'])
            # elif var == 'spread':
            #     if orderbook is None:
            #         state.append(0)
            #         continue
            #     
            #     spread = orderbook.get_ask() - orderbook.get_bid()
            #     
            #     if spread <= 1.:
            #         spread_discrete = 0
            #     elif spread > 2.:
            #         spread_discrete = 2
            #     else:
            #         spread_discrete = 1
            #     state.append(spread_discrete)
            else:
                val = orderbook.features[feat]


                if feat in ['high24hr', 'low24hr']:
                    val = val / orderbook.get_center()
                state.append(val)
                
            #else:
            #    raise NotImplemented

        return np.array(state)

    def get_action_index(self, action):
        action_idx = None
        assert action in self.actions, "Action not found: {} {}".format(action, self.actions)
        
        action_idx = self.actions.index(action)

        return action_idx

    def action_to_limit(self, action, init_center=None, current_ask=None):
        assert (self.limit_base == 'init_center' and isinstance(init_center, float)) or \
                (self.limit_base == 'curr_ask' and isinstance(current_ask, float))

        if self.limit_base == 'init_center':
            limit = init_center * (1. + (action/100.))
        elif self.limit_base == 'curr_ask':
            limit = current_ask * (1. + (action/100.))
        else:
            raise ValueError

        return limit


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

    def plot_evaluation_costs(self, experiments, vlines=[1], name=None, ylim=None, showfliers=False, hline=None):
        assert isinstance(hline, str) or hline is None
        
        experiments.plot.box(showmeans=True, color={'medians':'green'}, figsize=(12, 8), showfliers=showfliers)

        for line in vlines:
            plt.axvline(line + 0.5, color='black')
        # plt.axvline(experiments.shape[1]-0.5, color='black')
        plt.axhline(0, color='black')
        
        if hline is not None:
            if hline in experiments.columns:
                plt.axhline(experiments[hline].mean(), color='red', alpha=0.5, linewidth=2, linestyle='--', label='min mean')
                plt.axhline(experiments[hline].median(), color='green', alpha=0.5, linewidth=2, linestyle='--', label='min median')
            else:
                print("Could not find experiment '{}'".format(hline))

        title = "{} trading periods  \n{} - {}".format(len(experiments), experiments.index[0], experiments.index[-1])
        if name is not None:
            title = "{}: {}".format(name, title)
        
        plt.suptitle(title)
        plt.xlabel("Strategies")
        plt.ylabel("Occured costs")
        # plt.savefig("boxplot_train.pdf")
        plt.xticks(rotation=70)
        if ylim is not None:
            plt.ylim(ylim)    
        plt.legend(loc='best')
        
        plt.show()

    def evaluate(self, testdata, additional_agents=None, evaluate_actions=[], custom_strategies=None,
                 costs=None, name=None, which_min='first', verbose=False, show_plot=False, baseline=None):

        baseline = baseline or self.agent_name
        name = name or self.agent_name
        evaluate_agents = {
            self.agent_name: self
        }

        if additional_agents and isinstance(additional_agents, dict):
            evaluate_agents.update(additional_agents)

        if costs is not None:
            costs = costs
        else:
            costs = pd.DataFrame([])

        for w, window in tqdm_notebook(enumerate(testdata)):
            print(w)
            index = window[0].timestamp
            init_center = window[0].get_center()
            
            ## Learned strategy
            ots = OrderbookTradingSimulator(orderbooks=window, volume=self.V, tradingperiods=self.T,
                                             period_length=self.period_length)
            lim_increments = init_center * (self.lim_stepsize / 100)

            for agentname in evaluate_agents.keys():
                ots.reset()
                agent = evaluate_agents[agentname]

                for t in range(0, agent.T):
                    time_left = agent.T - t
                    timepoint = t*agent.period_length
                    
                    ob_now = window[timepoint]
                    price_incScale = int(round(ob_now.get_center()/lim_increments, 3))
                    
                    # agent.interpolate_vol = True
                    volume = float(ots.volume)
                    if hasattr(agent, 'volumes_base') and not agent.interpolate_vol:
                        # discretize volume through rounding (needed for QLookupTable)
                        volume = agent.round_custombase(volume, non_zero=True)
        
                    state = agent.generate_state(time_left=time_left, 
                                                volume_left=volume,
                                                orderbook=ob_now)

                    action, action_idx = agent.get_action(state, exploration=0, which_min=which_min)

                    if agent.limit_base == 'agression':
                        summary = ots.trade(agression_factor=action, extrainfo={'ACTION':action})
                    elif agent.limit_base == 'incStepUnits':
                        limit = lim_increments * (price_incScale + action)
                        summary = ots.trade(limit=limit, extrainfo={'ACTION':action})
                    else:
                        limit = agent.action_to_limit(action, init_center=init_center, current_ask=ob_now.get_ask())
                        summary = ots.trade(limit=limit, extrainfo={'ACTION':action})
                    # ots.trade(agression_factor=action, extrainfo={'ACTION':action})
                    # print(state, action, summary.costs.values[-1])

                    if ots.summary['done']:
                        break

                if verbose:
                    print(agentname, agent.limit_base, ots.history.cost.sum())
                    display(ots.history)

                costs.loc[index, agentname] = ots.history.cost.sum()

            if custom_strategies is not None:
                for strategyname in custom_strategies.keys():

                    strategy = custom_strategies[strategyname]
                    ots.reset()

                    for t in range(agent.T):
                        timepoint = t*self.period_length
                        ob_now = window[timepoint]
                        price_incScale = int(round(ob_now.get_center()/lim_increments, 3))

                        action = strategy[t]            
                        limit = lim_increments * (price_incScale + action)

                        summary = ots.trade(limit=limit, extrainfo={'ACTION':action})

                        if ots.summary['done']:
                            break

                    if verbose:
                        print(strategyname, agent.limit_base, ots.history.cost.sum())
                        display(ots.history)
                    costs.loc[index, strategyname] = ots.history.cost.sum()
            


                
            for action in evaluate_actions:
                ots.reset()
                limit = window[0].get_center() * (1. + (action/100.))

                price_incScale = int(round(window[0].get_center()/lim_increments, 3))
                limit = lim_increments * (price_incScale + action)

                for t in range(0, self.T):
                    ots.trade(limit = limit, extrainfo={'ACTION':action})

                    if ots.summary['done']:
                        break
                costs.loc[index, str(action)] = ots.history.cost.sum()

                if action==2 or action==6:
                    print(action, limit, ots.history.cost.sum())
                    display(ots.history)
                # if verbose:
                #     display(ots.history)
                
            if show_plot:
                clear_output(wait=True)
                vlines = [1]
                if additional_agents is not None:
                    vlines = vlines + [len(additional_agents)+1]
                if custom_strategies is not None:
                    vlines = vlines + [len(custom_strategies)+vlines[-1]]

                self.plot_evaluation_costs(
                    costs,
                    vlines=vlines,
                    hline=baseline,
                    showfliers=False)
                ev = costs.describe()
                ev.loc['rel_mean',:] = ev.loc['mean',:] / ev.loc['mean',baseline]
                ev.loc['rel_median',:] = ev.loc['50%',:] / ev.loc['50%',baseline]
                display(ev)
            
        return costs

    def sample_from_Q(self, vol_intervals, which_min):
        ''' this is only a skeleton '''
        raise NotImplementedE


    
    def heatmap_Q(self, hue='Q', vol_intervals=10, epoch=None, which_min='first', outfile=None, outformat='pdf', show_traces=False, show_minima_count=False):
        assert len(self.state_variables) == 2, "Not yet implemented for a statespace with more than 2 dimensions"
        
        if show_minima_count:
            fig, axs = plt.subplots(ncols=3, figsize=(16,4))
        else:
            fig, axs = plt.subplots(ncols=2, figsize=(16,5))

        df = self.sample_from_Q(vol_intervals=vol_intervals, which_min=which_min)
        
        sns.heatmap(df.pivot('time', 'volume', 'action'), annot=True, fmt="1.1f",
                    ax=axs[0], vmin=self.actions[0], vmax=self.actions[-1]) 
        axs[0].set_title('Optimal action')
        
        sns.heatmap(df.pivot('time', 'volume', 'q'), annot=True, annot_kws={'rotation':90}, fmt="1.2f", ax=axs[1])
        axs[1].set_title('Optimal Q value')
        
        if show_minima_count:
            sns.heatmap(df.pivot('time', 'volume', 'minima_count'), annot=True, ax=axs[2])
            axs[2].set_title('Minima Count')
        
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
        
        if outfile:
            if outfile[len(outformat):] != outformat:
                outfile = "{}.{}".format(outfile, outformat)
            plt.savefig(outfile, format=outformat)
        else:
            plt.show()
        plt.close()
