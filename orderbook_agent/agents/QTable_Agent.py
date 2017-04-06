import numpy as np
import pandas as pd
import json
import math
import random

from IPython.display import display

from .RL_Agent_Base import RLAgent_Base

class QTable_Agent(RLAgent_Base):

    def __init__(self, actions, vol_intervals, V=100, T=4, period_length=15,
                 agent_name='QTable_Agent', state_variables=['volume', 'time'],
                 normalized=True):
        super().__init__(
            actions=actions,
            V=V,
            T=T,
            period_length=period_length,
            agent_name=agent_name,
            state_variables=state_variables,
            normalized=normalized
        )

        self.vol_intervals = vol_intervals
        self.volumes = np.linspace(V, 0, num=self.vol_intervals+1)[:-1]
        self.volumes_base = V / self.vol_intervals
        self.q = {}
        self.n = {}  # n is the number of times we have tried an action in a state


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
               'agent_name': self.agent_name,
               'state_variables': self.state_variables,
               'normalized': self.normalized,
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
            agent_name=data['agent_name'],
            state_variables=data['state_variables'] or ['volume', 'time'],
            normalized=data['normalized']
            )
        
        ql.q = data['q']
        ql.n = data['n']
        for elem in ql.q:
            ql.q[elem] = np.array(ql.q[elem])
            ql.n[elem] = np.array(ql.n[elem])

        return ql

    def generate_state(self, time_left, volume_left, orderbook=None):
        '''
        returns a string version of the state
        '''

        return str(super().generate_state(time_left=time_left, volume_left=volume_left, orderbook=orderbook))

    def predict(self, state, action=None):
        Q_values = self.q.get(state, np.full(self.action_dim, np.nan))  # np.zeros(self.action_dim))
        
        if action:
            return Q_values[self.get_action_index(action)]

        return Q_values

    def getN(self, state, action=None):
        '''
        returns N: how often has an action been applied to a state
        '''
        N_values = self.n.get(state, np.zeros(self.action_dim))
        
        if action is not None:
            return N_values[self.get_action_index(action)]

        return N_values

    def learn(self, state, action, cost, new_state):

        oldv = self.q.get(state, None)
        
        action_idx = self.get_action_index(action)

        # get minQ from new_state
        if np.isnan(self.predict(new_state)).all():
            minQ_new_state = 0
        else:
            minQ_new_state = np.nanmin(self.predict(new_state))

        if oldv is None:
            self.q[state] = self.predict(state)
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

    def get_action(self, state, exploration=0):
        q = self.predict(state)

        if (np.isnan(q).all() == True):
            print("No Q-table entry found for state '{}'".format(state))
            return 0
        
        actions_idx = q.argmin()

        return self.actions[actions_idx], actions_idx

    def round_custombase(self, val, non_zero=False):
        rounded_volume = round(val / self.volumes_base) * self.volumes_base
        if non_zero:
            return max(rounded_volume, self.volumes_base)
        return rounded_volume

    def sample_from_Q(self, vol_intervals):
        assert len(self.state_variables) == 2, "Not yet implemented for more than 2 variables in state"

        df = pd.DataFrame([], columns=self.state_variables)
        for t in range(1, self.T+1):
            for v in self.volumes:
                state = self.generate_state(time_left=t,
                                            volume_left=v)
                
                q = self.predict(state)
                if not np.all([math.isnan(val) for val in q]):
                    action = self.actions[np.nanargmin(q)]
                    df_tmp = pd.DataFrame({'time': t,
                                           'volume': v,
                                           'q': np.nanmin(q),
                                           'state': state,
                                           'n': self.getN(state, action),
                                           'action': action}, index=["{},{:1.2f}".format(t, v)])
                    df = pd.concat([df, df_tmp])
        df['time'] = df.time.astype(int)
        return df
