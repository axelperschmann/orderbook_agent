import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import math
import datetime
from .orderbook_container import OrderbookContainer
import json
from IPython.display import display

import time

def round_custombase(val, base):
    return round(val / base) * base
        

class QLearn:

    def __init__(self, actions, vol_intervals, V=100, T=4, period_length=15, state_variables=['volume', 'time']):
        # assert isinstance(config, dict) or not config, "Config [if provided] must be of type 'dict', given: {}".format(type(config))
        # if not config:
        #     config = {}
        self.actions = actions
        self.actions_count = len(self.actions)
        self.vol_intervals = vol_intervals
        self.T = T
        self.V = V
        self.period_length = period_length
        self.state_variables = state_variables
        self.created = datetime.datetime.now()

        self.volumes = np.linspace(0, 1, num=self.vol_intervals+1)[1:]
        self.volumes_base = 1. / self.vol_intervals

        self.q = {}
        self.n = {}  # n is the number of times we have tried an action in a state

    def __str__(self):
        return("States: {}\nActions: {}".format(self.q, self.actions))

    def save(self, outfile):
        if outfile[-4:] != 'json':
            outfile = '{}.json'.format(outfile)

        puffer_q = {}
        puffer_n = {}
        for elem in self.q:
            puffer_q[elem] = self.q[elem].tolist()
            puffer_n[elem] = self.n[elem].tolist()

        obj = {'actions': self.actions,
               'vol_intervals': self.vol_intervals,
               'T': self.T,
               'V': self.V,
               'period_length': self.period_length,
               'state_variables': self.state_variables,
               'q': puffer_q,
               'n': puffer_n}

        with open(outfile, 'w') as f_out:
            f_out.write(json.dumps(obj) + "\n")

        print("Saved: '{}'".format(outfile))

    def load(infile):
        if infile[-4:] != 'json':
            infile = '{}.json'.format(infile)
        
        with open(infile, 'r') as f:
            data = json.load(f)
        
        ql = QLearn(
            actions=data['actions'],
            vol_intervals=data['vol_intervals'],
            T=data['T'],
            V=data['V'],
            period_length=data['period_length'],
            state_variables=data['state_variables'] or ['volume', 'time']
            )
        
        ql.q = data['q']
        ql.n = data['n']
        for elem in ql.q:
            ql.q[elem] = np.array(ql.q[elem])
            ql.n[elem] = np.array(ql.n[elem])

        return ql
    
    def merge_Qlearners(self, other, inplace=False):
        # assert matching training settings
        assert self.actions == other.actions
        assert self.vol_intervals == other.vol_intervals
        assert self.T == other.T
        assert self.V == other.V
        assert self.period_length == other.period_length
        assert self.state_variables == other.state_variables
        
        if inplace:
            ql = self
        else:
            ql = QLearn(
                actions=self.actions,
                vol_intervals=self.vol_intervals,
                T=self.T,
                V=self.V,
                period_length=self.period_length,
                state_variables=self.state_variables
            )
        
        for state in self.n:
            new_n = self.n[state] + other.n[state]
            ql.q[state] = (self.q[state] * self.n[state] + other.q[state] * other.n[state]) / new_n
            ql.n[state] = new_n
        
        if not inplace:
            return ql
    
    def state_as_string(self, time_left, volume_left, orderbook=None):   
        assert isinstance(time_left, int), "Parameter 'time_left' must be of type 'int', given: '{}'".format(type(time_left))
        assert isinstance(volume_left, (int, float)), "Parameter 'volume_left' must be of type 'int' or 'float', given: '{}'".format(type(volume_left))
        assert isinstance(orderbook, OrderbookContainer ) or orderbook is None, "Parameter 'orderbook' [if provided] must be of type 'Orderbook', given: '{}'".format(type(orderbook))

        allowed_variable_set = ['volume', 'time', 'spread']
        assert set(self.state_variables).issubset(allowed_variable_set), "Parameter 'state_variables' must be a subset of {}".format(allowed_variable_set)

        state = []
        for var in self.state_variables:
            if var == 'volume':
                state.append("{:1.2f}".format(volume_left))
            elif var == 'time':
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

        return str(state)

    def get_action_index(self, action):
        action_idx = None
        assert action in self.actions, "Action not found: {}".format(action)
        
        action_idx = self.actions.index(action)

        return action_idx

    def getQ(self, state, action=None):

        Q_values = self.q.get(state, np.full(self.actions_count, np.nan))  # np.zeros(self.actions_count))
        
        if action:
            return Q_values[self.get_action_index(action)]

        return Q_values

    def getN(self, state, action=None):
        # how often has an action been applied to a state?!
        N_values = self.n.get(state, np.zeros(self.actions_count))
        
        if action is not None:
            return N_values[self.get_action_index(action)]

        return N_values

    def learn(self, state, action, cost, new_state):
        oldv = self.q.get(state, None)
        
        action_idx = self.get_action_index(action)

        # get minQ from new_state
        if np.isnan(self.getQ(new_state)).all():
            minQ_new_state = 0
        else:
            minQ_new_state = np.nanmin(self.getQ(new_state))

        if oldv is None:
            self.q[state] = self.getQ(state)
            self.q[state][action_idx] = cost + minQ_new_state

            init_n = self.getN(state)
            init_n[action_idx] = 1
            self.n[state] = init_n

        else:
            n = self.getN(state, action=action)
            
            if n == 0:
                self.q[state][action_idx] = (cost + minQ_new_state)
            else:
                self.q[state][action_idx] = n/(n+1) * self.q[state][action_idx] + 1/(n+1) * (cost + minQ_new_state)

            self.n[state][action_idx] = n + 1

    def chooseAction(self, state):
        q = self.getQ(state)
        if not (np.isnan(q).all() == False):
            return 0
        assert np.isnan(q).all() == False, "q table is empty for state '{}'. Probably a missmatch between parameter 'V' used for training and used now.".format(state) 
        
        min_indices = np.where(q == q.min())[0]
        
        if len(min_indices) > 1:
            i = random.choice(min_indices)
        else:
            i = min_indices[0]
        
        
        return self.actions[i]
    
    def heatmap_Q(self, hue='Q', epoch=None, outfile=None, outformat='pdf'):
        
        assert len(self.state_variables) == 2, "Not yet implemented for more than 2 variables in state"
        
        df = pd.DataFrame([], columns=self.state_variables)
        for t in range(1, self.T+1):
            for v in self.volumes[::-1]:
                state = self.state_as_string(time_left=t,
                                        volume_left=v)
                q = self.getQ(state)
                if not np.all([math.isnan(val) for val in q]):
                    action = self.actions[np.nanargmin(q)]
                    df_tmp = pd.DataFrame({'time': int(t),
                                           'volume': v,
                                           'q': np.nanmin(q),
                                           'n': self.getN(state, action),
                                           'action': action}, index=["{},{:1.2f}".format(t, v)])
                    df = pd.concat([df, df_tmp])
        df['time'] = df.time.astype(int)

        fig, axs = plt.subplots(ncols=2, figsize=(16,5))
        plt.suptitle="X"
        sns.heatmap(df.pivot('time', 'volume', 'action'), annot=True, fmt="1.2f",
                    ax=axs[0], vmin=-0.4, vmax=1.0)
        sns.heatmap(df.pivot('time', 'volume', 'q'), annot=True, fmt="1.2f", ax=axs[1])
        title = "Q function (T:{}, V:{})".format(self.T*self.period_length, self.V)
        if epoch is not None:
            title = "{}, epochs:{}".format(title, epoch+1)
            
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

        

    def plot_Q(self, z_represents='action', epoch=None, outfile=None, outformat='pdf', verbose=False):
        assert isinstance(z_represents, str) and z_represents in ['action', 'Q']
        
        timesteps = range(1, self.T+1)
        grey_tones = np.linspace(1, 0.3, num=self.T)
        
        fig = plt.figure()
        ax = [fig.add_subplot(111, projection='3d')]
        
        length = self.T * self.vol_intervals

        x_offset = 0.5
        y_offset = 0.5*self.volumes_base

        xpos = []
        ypos = []
        zpos = [0] * length

        dx = np.ones(length) / self.vol_intervals 
        dy = np.ones(length)
        dz_action = []
        dz_q = []

        colors = []
        
        for t in timesteps:
            for v in self.volumes:
                # state = '[{}, {:1.1f}]'.format(t, v)
                state = self.state_as_string(time_left=t, volume_left=v)
                
                xpos.append(t-x_offset)
                ypos.append(v-y_offset)
                colors.append((grey_tones[t-1], grey_tones[t-1], grey_tones[t-1]))
                
                q = self.getQ(state)

                if np.all([math.isnan(val) for val in q]):
                    xpos.pop()
                    ypos.pop()
                    zpos.pop()
                    colors.pop()
                    dx = dx[:-1]
                    dy = dy[:-1]
                else:
                    dz_action.append(self.actions[np.nanargmin(q)])
                    dz_q.append(np.nanmin(q))

        if z_represents == 'action':
            ax[0].bar3d(ypos, xpos, zpos, dx, dy, dz=dz_action, color=colors, alpha=1)
            ax[0].set_zlabel("optimal action")
        elif z_represents == 'Q':
            ax[0].bar3d(ypos, xpos, zpos, dx, dy, dz=dz_q, color=colors, alpha=1)
            ax[0].set_zlabel("Q Value")

        # layout
        for axis in ax:
            axis.set_ylabel("time remaining [periods]")
            axis.set_xlim3d((y_offset,self.volumes[-1]+y_offset))
            axis.invert_xaxis()
            
            axis.set_ylim3d((x_offset,self.T+x_offset))
            axis.set_yticks(timesteps)
            axis.invert_yaxis()
            
            axis.set_xlabel("trade volume remaining [%]")
            axis.set_xticks(self.volumes)
            plt.tight_layout()       

        title = "Q function (T:{}, V:{})".format(self.T*self.period_length, self.V)
        if epoch is not None:
            title = "{}, epochs:{}".format(title, epoch+1)
        fig.suptitle(title)

        if outfile:
            if outfile[len(outformat):] != outformat:
                outfile = "{}.{}".format(outfile, outformat)
            plt.savefig(outfile, format=outformat)

            if verbose:
                print("Successfully saved '{}'".format(outfile))
        else:
            plt.show()
        plt.close()
