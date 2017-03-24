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

# Neural Network
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

def base_model(input_dim=2, output_dim=15):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu', init='glorot_normal'))
    model.add(Dense(output_dim, activation='linear', init='glorot_normal'))
    model.compile(loss='mse', optimizer='Adam')
    return model
        

class RLAgent:

    def __init__(self, actions, model=None, V=100, T=4, period_length=15, state_variables=['volume', 'time']):
        self.actions = actions
        self.state_dim = len(state_variables)
        self.action_dim = len(actions)

        assert isinstance(model, Sequential) or model is None, "Given: '{}'".format(type(model))
        if model is not None:
            assert model.input_shape[1] == self.state_dim
            assert model.model.output_shape[1] == self.action_dim
        self.model = model or base_model(input_dim=self.state_dim, output_dim=self.action_dim)
    
        self.T = T
        self.V = V
        self.period_length = period_length
        self.state_variables = state_variables
        self.created = datetime.datetime.now()


        self.q = {}
        self.n = {}  # n is the number of times we have tried an action in a state

    def __str__(self):
        return("States: {}\nActions: {}".format(self.q, self.actions))

    def save(self, outfile):
        # ToDo: Adapt to RLAgent
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
        # ToDo: Adapt to RLAgent
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
    
    def generate_state(self, time_left, volume_left, orderbook=None):   
        assert isinstance(time_left, (int, float)), "Parameter 'time_left' must be of type 'int', given: '{}'".format(type(time_left))
        assert isinstance(volume_left, (int, float)), "Parameter 'volume_left' must be of type 'int' or 'float', given: '{}'".format(type(volume_left))
        assert isinstance(orderbook, OrderbookContainer ) or orderbook is None, "Parameter 'orderbook' [if provided] must be of type 'Orderbook', given: '{}'".format(type(orderbook))

        allowed_variable_set = ['volume', 'time', 'spread']
        assert set(self.state_variables).issubset(allowed_variable_set), "Parameter 'state_variables' must be a subset of {}".format(allowed_variable_set)

        state = []
        for var in self.state_variables:
            if var == 'volume':
                state.append(volume_left/self.V)
            elif var == 'time':
                state.append(time_left/self.T)
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

    def predict(self, state):

        return self.model.predict(state.reshape(1, self.state_dim))

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
    
    def heatmap_Q(self, hue='Q', vol_intervals=10, epoch=None, outfile=None, outformat='pdf'):
        
        assert len(self.state_variables) == 2, "Not yet implemented for more than 2 variables in state"
        print("T", self.T)
        df = pd.DataFrame([], columns=self.state_variables)
        for t in range(1, self.T+1):
            for v in np.linspace(0, self.V, num=vol_intervals+1)[1:]:
                state = self.generate_state(time_left=t,
                                        volume_left=v)
                print(state)
                q = self.predict(state)

                if t==1 and v==100:
                    print("t1, v100", q)
                elif t==1 and v==20:
                    print("t1, v20 ", q)
                
                action = self.actions[np.nanargmin(q)]
                df_tmp = pd.DataFrame({'time': t,
                                       'state': str(state),
                                       'volume': v,
                                       'q': np.nanmin(q),
                                       'action': action}, index=["{:1.2f},{}".format(v, t)])
                df = pd.concat([df, df_tmp])
        # df['time'] = df.time.astype(int)

        fig, axs = plt.subplots(ncols=2, figsize=(16,5))
        plt.suptitle="X"
        display(df)
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
