import numpy as np
import pandas as pd
import json
import math
import random
import os
from tqdm import tqdm, tqdm_notebook
from IPython.display import display

from .RL_Agent_Base import RLAgent_Base
from helper.general_helpers import safe_list_get


class QTable_Agent(RLAgent_Base):

    def __init__(self, actions, vol_intervals, limit_base, V=100, T=4, consume='volume',
                 period_length=15, samples=None, lim_stepsize=0.1,
                 agent_name='QTable_Agent', state_variables=['volume', 'time'],
                 normalized=True, interpolate_vol=False, agent_type='QTable_Agent'):
        super().__init__(
            actions=actions,
            lim_stepsize=lim_stepsize,
            V=V,
            T=T,
            consume=consume,
            period_length=period_length,
            samples=samples,
            agent_name=agent_name,
            state_variables=state_variables,
            normalized=normalized,
            limit_base=limit_base,
            agent_type=agent_type
        )

        self.vol_intervals = vol_intervals
        self.volumes = np.linspace(V, 0, num=self.vol_intervals+1)[:-1]
        self.volumes_base = V / self.vol_intervals
        self.interpolate_vol = interpolate_vol
        self.q = {}
        self.n = {}  # n is the number of times we have tried an action in a state


    def convert_to_BatchTreeAgent(self, new_name, train=False):
        from .BatchTree_Agent import RLAgent_BatchTree
        params = self.get_params()
        del(params['n'], 
            params['q'], 
            params['vol_intervals'], 
            params['interpolate_vol'], 
            params['agent_type']
            )
        new_agent = RLAgent_BatchTree(**params)
        new_agent.agent_name = new_name
        new_agent.samples = self.samples.copy()

        if train:
            new_agent.learn_fromSamples()

        return new_agent

    def get_params(self):
        puffer_q = {}
        puffer_n = {}
        for elem in self.q:
            puffer_q[elem] = self.q[elem].tolist()
            puffer_n[elem] = self.n[elem].tolist()
        params = {'actions': self.actions,
               'lim_stepsize': self.lim_stepsize,
               'vol_intervals': self.vol_intervals,
               'V': self.V,
               'T': self.T,
               'consume': self.consume,
               'period_length': self.period_length,
               'agent_name': self.agent_name,
               'state_variables': self.state_variables,
               'normalized': self.normalized,
               'limit_base': self.limit_base,
               'interpolate_vol': self.interpolate_vol,
               'q': puffer_q,
               'n': puffer_n,
               'agent_type': 'QTable_Agent',
               }

        return params

    def copy(self, new_name=None):
        params = self.get_params()
        q = params.pop('q')
        n = params.pop('n')

        new_agent = QTable_Agent(**params)
        if isinstance(new_name, str):
            new_agent.agent_name = new_name

        new_agent.samples = self.samples.copy()
        new_agent.q = q
        new_agent.n = n

        return new_agent


    def save(self, path=".", outfile_agent=None, outfile_samples=None, overwrite=False):
        if outfile_agent is None:
            outfile_agent = self.agent_name
        if outfile_samples is None:
            outfile_samples = self.agent_name

        # append file type
        if outfile_agent.split(".")[-1] != 'json':
            outfile_agent = '{}.json'.format(outfile_agent)
        if outfile_samples.split(".")[-1] != 'csv':
                outfile_samples = '{}.csv'.format(outfile_samples)

        obj = self.get_params()

        if not os.path.exists(path):
            os.makedirs(path)

        # save agent to disk
        outfile_agent = os.path.join(path, outfile_agent)
        if os.path.isfile(outfile_agent) and overwrite is False:
            print("File '{}' exists! Do not overwrite!".format(outfile_agent))
        else:
            with open(outfile_agent, 'w') as f_out:
                f_out.write(json.dumps(obj, default=lambda df: json.loads(df.to_json())) + "\n")
            print("Saved agent: '{}'".format(outfile_agent))

        # save samples to disk
        outfile_samples = os.path.join(path, outfile_samples)
        if os.path.isfile(outfile_samples) and overwrite is False:
            print("File '{}'  exists! Do not overwrite!".format(outfile_samples))
        else:
            self.samples.to_csv(outfile_samples)
            print("Saved samples: '{}'".format(outfile_samples))

    def load(agent_name=None, path='.', infile_agent=None, infile_samples=None, ignore_samples=False):
        if agent_name is None:
            assert isinstance(infile_agent, str), "Bad parameter 'infile_agent', given: {}".format(infile_agent)
            assert isinstance(infile_samples, str), "Bad parameter 'infile_samples', given: {}".format(infile_samples)
        else:
            infile_agent = "{}.json".format(agent_name)
            infile_samples = "{}.csv".format(agent_name)


        with open(os.path.join(path, infile_agent), 'r') as f:
            data = json.load(f)
        
        ql = QTable_Agent(
            actions=data['actions'],
            lim_stepsize=safe_list_get(data, idx='lim_stepsize', default=0.1),
            vol_intervals=data['vol_intervals'],
            V=data['V'],
            T=data['T'],
            consume=safe_list_get(data, idx='consume', default='volume'),
            period_length=data['period_length'],
            agent_name=data['agent_name'],
            state_variables=data['state_variables'] or ['volume', 'time'],
            normalized=data['normalized'],
            limit_base=data['limit_base'],
            interpolate_vol=data['interpolate_vol']
            )
        
        ql.q = data['q']
        ql.n = data['n']
        for elem in ql.q:
            ql.q[elem] = np.array(ql.q[elem])
            ql.n[elem] = np.array(ql.n[elem])

        if ignore_samples:
            ql.samples = pd.DataFrame()
            print("No samples loaded! Parameter 'ignore_samples'==True")
        else:
            ql.samples = pd.read_csv(os.path.join(path, infile_samples), parse_dates=['timestamp'], index_col=0)
            ql.samples['timestamp']

        return ql

    def learn_fromSamples(self, reset_brain=True, verbose=False):
        if reset_brain:
            self.q = {}
            self.n = {}

        for tt in tqdm(range(self.T)[::-1]):
            time_left = self.T-tt
            
            df_sub = self.samples[self.samples.time==time_left]
            
            for idx, row in enumerate(df_sub.iterrows()):
                if verbose:
                    if idx%5000==0:
                        print("{}/{}".format(idx, len(df_sub)))

                sample = row[1]

                state = sample[self.state_variables].copy()
                assert pd.isnull(state).any()==False, "NaN in state: '{}', '{}'\n{}".format(state.values, state.index, sample)

                state['volume'] = self.round_custombase(state.volume)
                # print("vol", state.values)
                if self.normalized:
                    state['volume'] = state['volume'] / self.V
                    state['time'] = state['time'] / self.T
                state = list(state.values)
                
                new_state = sample[["{}_n".format(var) for var in self.state_variables]]
                new_state['volume_n'] = self.round_custombase(new_state.volume_n)
                if self.normalized:
                    new_state['volume_n'] = new_state['volume_n'] / self.V
                    new_state['time_n'] = new_state['time_n'] / self.T

                new_state = list(new_state.values)

                volume_traded = sample.volume-sample.volume_n
                volume_traded_rounded = self.round_custombase(volume_traded)

                if volume_traded == 0:
                    cost = 0
                else:
                    cost = sample.cost / volume_traded * volume_traded_rounded

                # print("{}   {:1.2f}, {:1.4f}   {}".format(state, sample.action, cost, new_state))
                self.learn(state=state, action=sample.action, cost=cost, new_state=new_state)

    def generate_state(self, time_left, volume_left, round_vol=False, orderbook=None, orderbook_cheat=None, extra_variables=None):
        if round_vol:  # ToDo: Remove end of line: and not self.interpolate_vol:
            # round volume for proper Table Lookup
            volume_left = self.round_custombase(volume_left, non_zero=True)

        return (super().generate_state(time_left=time_left, volume_left=volume_left, orderbook=orderbook, orderbook_cheat=orderbook_cheat, extra_variables=extra_variables))

    def _get_nearest_neighbours(self, state, variable='volume'):
        assert isinstance(variable, str) and variable in ['volume']
        volumes = self.volumes
        if 0. not in volumes:
            volumes = np.array(list(volumes) + [0])

        if variable == 'volume':
            vol = state[0]
            # find nearest neighbours
            if self.normalized:
                idx = np.argsort(abs(volumes/self.V-vol))[:2]
            else:
                idx = np.argsort(abs(volumes-vol))[:2]

            neighbours = volumes[sorted(idx)]

            if self.normalized:
                w = (vol - (neighbours[1]/self.V))  / (self.volumes_base/self.V)
            else:
                w = (vol/self.V - (neighbours[1]/self.V))  / (self.volumes_base/self.V)
            
            n1 = state.copy()
            n2 = state.copy()
            n1[0] = neighbours[0]
            n2[0] = neighbours[1]

            if self.normalized:
                n1[0] /= self.V
                n2[0] /= self.V

        return n1, n2, w


    def predict(self, state, action=None):
        Q_values = self.q.get(str(state), np.full(self.action_dim, np.nan))  # np.zeros(self.action_dim))
        if self.interpolate_vol:
            
            if str(state) not in self.q.keys():
                # only interpolate between states when necessary
                vol = state[0]
                if vol>self.volumes_base/self.V:
                    # interpolate between neighboring states (volume based neigbours)
                    n1, n2, w = self._get_nearest_neighbours(state, variable='volume')

                    pred1 = self.q.get(str(n1), np.full(self.action_dim, np.nan))
                    pred2 = self.q.get(str(n2), np.full(self.action_dim, np.nan))

                    if np.all([math.isnan(val) for val in pred2]):
                        pred2 = np.array([0]*self.action_dim)
                    
                    if not(np.all([math.isnan(val) for val in pred1]) or np.all([math.isnan(val) for val in pred2])):
                        Q_values = w*pred1 + (1-w)*pred2
                else:
                    nearest_state = state.copy()
                    nearest_state[0] = self.volumes_base/self.V
                    # print("nearest state:", state, nearest_state)
                    Q_values = self.q.get(str(nearest_state), np.full(self.action_dim, np.nan))

        if action:
            return Q_values[self.get_action_index(action)]

        return Q_values

    def getN(self, state, action=None):
        '''
        returns N: how often has an action been applied to a state
        '''
        N_values = self.n.get(str(state), np.zeros(self.action_dim))
        
        if action is not None:
            return N_values[self.get_action_index(action)]

        return N_values

    def learn(self, state, action, cost, new_state):
        state = [float(v) for v in state]

        oldv = self.q.get(str(state), None)
        
        action_idx = self.get_action_index(action)

        # get minQ from new_state
        new_state_v = self.predict(new_state)
        if np.isnan(new_state_v).all():
            minQ_new_state = 0
        else:
            minQ_new_state = np.nanmin(new_state_v)

        if oldv is None:
            self.q[str(state)] = self.predict(state)
            self.q[str(state)][action_idx] = cost + minQ_new_state
        
            init_n = self.getN(state)
            init_n[action_idx] = 1
            self.n[str(state)] = init_n

        else:
            n = self.getN(state, action=action)
            
            if n == 0:
                self.q[str(state)][action_idx] = (cost + minQ_new_state)
            else:
                self.q[str(state)][action_idx] = n/(n+1) * self.q[str(state)][action_idx] + 1/(n+1) * (cost + minQ_new_state)

            self.n[str(state)][action_idx] = n + 1

    def get_action(self, state, which_min='first'):
        assert isinstance(which_min, str) and which_min in ['first', 'last']

        q = np.array(self.predict(state))

        if (np.isnan(q).all() == True):
            print("No Q-table entry found for state '{}'".format(state))
            q = np.zeros_like(q)
            # raise ValueError
        
        if which_min == 'first':
            # if multiple minima exist, choose first one (lowest aggression level)
            action_idx = np.nanargmin(q)
        elif which_min == 'last':
            # if multiple minima exist, choose last one (highest aggression level)
            action_idx = np.where(q == np.nanmin(q))[0][-1]

        return self.actions[action_idx], action_idx

    def round_custombase(self, val, non_zero=False):
        rounded_volume = round(val / self.volumes_base) * self.volumes_base
        
        if non_zero:
            return max(rounded_volume, self.volumes_base)
        return rounded_volume

    def sample_from_Q(self, vol_intervals, which_min, extra_variables=None):
        # assert len(self.state_variables) == 2, "Not yet implemented for more than 2 variables in state"
        
        if self.interpolate_vol:
            volumes = np.linspace(self.V, 0, num=vol_intervals+1)[:-1] 
        else:
            volumes = self.volumes

        df = pd.DataFrame([], columns=self.state_variables)
        
        for t in range(1, self.T+1):
            for v in volumes:
                state = self.generate_state(time_left=t,
                                            volume_left=v,
                                            extra_variables=extra_variables)
                q = np.array(self.predict(state))
                if not np.all([math.isnan(val) for val in q]):
                    action, action_idx = self.get_action(state, which_min=which_min)
                    minima_count = len(np.where(q == np.nanmin(q))[0])

                    df_tmp = pd.DataFrame({'time': t,
                                           'volume': v,
                                           'q': np.nanmin(q),
                                           'state': str(state),
                                           'n': self.getN(state, action),
                                           'minima_count': minima_count,
                                           'action': action}, index=["{},{:1.2f}".format(t, v)])
                    df = pd.concat([df, df_tmp])
                else:
                    print('q', q)
        df['time'] = df.time.astype(int)
        return df
