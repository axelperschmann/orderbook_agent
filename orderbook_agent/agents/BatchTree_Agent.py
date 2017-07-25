from tqdm import tqdm, tqdm_notebook
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import random
import dill
import json
import os

from .RL_Agent_Base import RLAgent_Base
from helper.orderbook_trader import OrderbookTradingSimulator
from helper.general_helpers import safe_list_get

class RLAgent_BatchTree(RLAgent_Base):
    def __init__(self, actions, V, T, consume, period_length, limit_base,
                 model=None, samples=None, agent_name='BatchTree_Agent', lim_stepsize=0.1,
                 state_variables=['volume', 'time'], normalized=False, agent_type='BatchTree_Agent'):
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
            agent_type=agent_type)
        self.model = model

    def get_params(self):
        params = {
               'actions': self.actions,
               'lim_stepsize': self.lim_stepsize,
               'V': self.V,
               'T': self.T,
               'consume': self.consume,
               'period_length': self.period_length,
               'agent_name': self.agent_name,
               'state_variables': self.state_variables,
               'normalized': self.normalized,
               'limit_base': self.limit_base,
               'agent_type': 'BatchTree_Agent',
               }

        return params

    def copy(self, new_name=None):
        params = self.get_params()

        new_agent = RLAgent_BatchTree(**params)
        if isinstance(new_name, str):
            new_agent.agent_name = new_name
        new_agent.samples = self.samples.copy()
        new_agent.model = self.model

        return new_agent

    def save(self, path=".", outfile_agent=None, outfile_model=None, outfile_samples=None, overwrite=False, ignore_samples=False):
        if outfile_agent is None:
            outfile_agent = self.agent_name
        if outfile_model is None:
            outfile_model = self.agent_name
        if outfile_samples is None:
            outfile_samples = self.agent_name

        # append file type
        if outfile_agent.split(".")[-1] != 'json':
            outfile_agent = '{}.json'.format(outfile_agent)
        if outfile_model.split(".")[-1] != 'model':
            outfile_model = '{}.model'.format(outfile_model)
        if outfile_samples.split(".")[-1] != 'csv':
            outfile_samples = '{}.csv'.format(outfile_samples)

        obj = self.get_params()

        if not os.path.exists(path):
            os.makedirs(path)

        ### save agent to disk
        outfile_agent = os.path.join(path, outfile_agent)
        if os.path.isfile(outfile_agent) and overwrite is False:
            print("File '{}' exists! Do not overwrite!".format(outfile_agent))
        else:
            with open(outfile_agent, 'w') as f_out:
                f_out.write(json.dumps(obj, default=lambda df: json.loads(df.to_json())) + "\n")
            print("Saved agent: '{}'".format(outfile_agent))

        ### save agent model to disk
        outfile_model = os.path.join(path, outfile_model)
        if os.path.isfile(outfile_model) and overwrite is False:
            print("File '{}' exists! Do not overwrite!".format(outfile_model))
        else:
            with open(outfile_model, 'wb') as f_out:
                dill.dump(self.model, f_out)
            print("Saved model: '{}'".format(outfile_model))

        ### save samples to disk
        if ignore_samples:
            print("ignoring samples")
        else:
            outfile_samples = os.path.join(path, outfile_samples)
            if os.path.isfile(outfile_samples) and overwrite is False:
                print("File '{}'  exists! Do not overwrite!".format(outfile_samples))
            else:
                self.samples.to_csv(outfile_samples)
                print("Saved samples: '{}'".format(outfile_samples))

    def load(agent_name=None, path='.', infile_agent=None, infile_model=None, infile_samples=None, ignore_samples=False):
        if agent_name is None:
            assert isinstance(infile_agent, str), "Bad parameter 'infile_agent', given: {}".format(infile_agent)
            assert isinstance(infile_model, str), "Bad parameter 'infile_model', given: {}".format(infile_model)
            assert isinstance(infile_samples, str), "Bad parameter 'infile_samples', given: {}".format(infile_samples)
        else:
            infile_agent = "{}.json".format(agent_name)
            infile_model = "{}.model".format(agent_name)
            infile_samples = "{}.csv".format(agent_name)

        with open(os.path.join(path, infile_agent), 'r') as f:
            data = json.load(f)
        
        ql = RLAgent_BatchTree(
            actions=data['actions'],
            lim_stepsize=safe_list_get(data, idx='lim_stepsize', default=0.1),
            V=data['V'],
            T=data['T'],
            consume=safe_list_get(data, idx='consume', default='volume'),
            period_length=data['period_length'],
            agent_name=data['agent_name'],
            state_variables=data['state_variables'] or ['volume', 'time'],
            normalized=data['normalized'],
            limit_base=data['limit_base'],
            )

        if ignore_samples:
            ql.samples = pd.DataFrame()
            # print("No samples loaded! Parameter 'ignore_samples'==True")
        else:
            ql.samples = pd.read_csv(os.path.join(path, infile_samples), parse_dates=['timestamp'], index_col=0)
        ql.model = dill.load(open(os.path.join(path, infile_model), "rb"))

        return ql

    def predict(self, states):
        assert self.model is not None, "Agent has no model yet. Call 'learn_fromSamples() to start training from {} samples".format(self.samples.shape)

        df = pd.DataFrame(states).T
        
        preds = []
        for a in self.actions:
            d = pd.concat([df, pd.DataFrame(np.ones(df.shape[0])*a)], axis=1)
            preds.append(self.model.predict(d)[0])

        return preds

    def get_action(self, p_feat, which_min='first'):
        assert isinstance(which_min, str) and which_min in ['first', 'last']

        if self.model is None:
            action_idx = random.randint(0, len(self.actions)-1)
        else:   
            preds = np.array(self.predict(p_feat))

            if which_min == 'first':
                # if multiple minima exist, choose first one (lowest aggression level)
                action_idx = np.nanargmin(preds)
            elif which_min == 'last':
                # if multiple minima exist, choose last one (highest aggression level)
                action_idx = np.where(preds == np.nanmin(preds))[0][-1]
            
        return self.actions[action_idx], action_idx
        
    def learn(self, state, action, cost, new_state):
        '''
        look for function instead: learn_fromSamples()
        '''
        print("learn")
        raise NotImplementedError    

    def learn_fromSamples(self, nb_it=None, n_estimators=20, max_depth=12, verbose=False):
        if nb_it is None:
            nb_it = self.T

        reg = None
        d_rate = 0.95
        df = self.samples.copy()

        df.insert(loc=len(df.columns)-self.state_dim, column='done', value=abs(df['volume_n']) < 0.00001)
        df.insert(loc=len(df.columns)-self.state_dim, column='min_cost', value=df['cost'])
        df['min_cost'] = df['cost']

        if self.state_variables_actions != False:
            state_variables = self.state_variables_actions
        else:
            state_variables = self.state_variables
        print("x")
        for n in tqdm(range(nb_it)):
            if verbose:
                print("n", n)

            # training an estimate of q_1
            if reg is not None:
                # using previous classifier as estimate of q_n-1
                states = df[["{}_n".format(var) for var in state_variables]].copy()

                
                preds = []

                for a in self.actions:
                    if verbose:
                        print("a", a)
                    states['action'] = a
                    # display(a, states.shape, states.dropna().shape)
                    preds.append(reg.predict(states))
                preds = pd.DataFrame(preds, index=self.actions, columns=df.index)
                
                # preparing our new training data set
                # if done, do not add any future costs anymore.
                df['min_cost'] = (df['cost'] + (1 - df['done']) * (preds.min() * d_rate))
                
            reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
            reg = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth)
            print("fit")
            X = df[state_variables+['action']]
            y = df['min_cost']
            display(X.head())
            display(y.head())
            reg = reg.fit(X, y)
            print("fit done")
            if verbose:
                print("Score:", reg.score(df[state_variables+['action']], df['min_cost']))
                print("Feature importances:", reg.feature_importances_)
            self.model = reg

            if len(state_variables) == 2:
                self.heatmap_Q(
                    show_traces=False,
                    show_minima_count=False,
                    vol_intervals=24)
            if (len(state_variables) == 3) and ('spread' in state_variables):
                self.heatmap_Q(
                    show_traces=False,
                    show_minima_count=False,
                    vol_intervals=24,
                    extra_variables={'spread': 0.001845},
                    outfile='{}_{}'.format(self.agent_name, n)
                    )

    def sample_from_Q(self, vol_intervals, which_min, extra_variables=None):
        # assert len(self.state_variables) == 2, "Not yet implemented for more than 2 variables in state"
        
        df = pd.DataFrame([])
        for t in range(1, self.T+1):
            for v in np.linspace(0, self.V, num=vol_intervals+1)[1:]:
                
                state = self.generate_state(time_left=t,
                                        volume_left=v,
                                        extra_variables=extra_variables)

                q = np.array(self.predict(state))

                action, action_idx = self.get_action(state, which_min=which_min)
                minima_count = len(np.where(q == np.nanmin(q))[0])
                
                df_tmp = pd.DataFrame({'time': t,
                                       'state': str(state),
                                       'volume': v,
                                       'q': np.nanmin(q),
                                       'minima_count': minima_count,
                                       'action': action}, index=["{:1.2f},{}".format(v, t)])
                df = pd.concat([df, df_tmp])

        return df
