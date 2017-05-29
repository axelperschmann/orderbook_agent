from tqdm import tqdm, tqdm_notebook
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor
import random

from .RL_Agent_Base import RLAgent_Base
from helper.orderbook_trader import OrderbookTradingSimulator


class RLAgent_BatchTree(RLAgent_Base):
    def __init__(self, actions, lim_stepsize, V, T, consume, period_length, limit_base,
                 model=None, samples=None, agent_name='BatchTree_Agent',
                 state_variables=['volume', 'time'], normalized=False):
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
            limit_base=limit_base)
        self.model = model

    def predict(self, states):
        df = pd.DataFrame(states).T
        
        preds = []
        for a in self.actions:
            d = pd.concat([df, pd.DataFrame(np.ones(df.shape[0])*a)], axis=1)
            preds.append(self.model.predict(d)[0])

        return preds

    def get_action(self, p_feat, exploration=0, which_min='first'):
        assert isinstance(which_min, str) and which_min in ['first', 'last']
        
        if self.model is None or random.random() < exploration:
            action_idx = random.randint(0, len(self.actions)-1)
            action = self.actions[action_idx]
        else:   
            preds = np.array(self.predict(p_feat))

            if which_min == 'first':
                # if multiple minima exist, choose first one (lowest aggression level)
                action_idx = np.nanargmin(preds)
            elif which_min == 'last':
                # if multiple minima exist, choose last one (highest aggression level)
                action_idx = np.where(preds == np.nanmin(preds))[0][-1]
            action = self.actions[action_idx]
            
        return action, int(action_idx)
        
    def learn(self, state, action, cost, new_state):
        '''
        look for function instead: fitted_Q_iteration_tree()
        '''
        print("learn")
        raise NotImplementedError    

    def fitted_Q_iteration_tree(self, nb_it, n_estimators=20, max_depth=12, verbose=False):
        reg = None
        d_rate = 0.95
        df = self.samples.copy()

        df.insert(loc=len(df.columns)-self.state_dim, column='done', value=abs(df['volume_n']) < 0.00001)
        df.insert(loc=len(df.columns)-self.state_dim, column='min_cost', value=df['cost'])
        df['min_cost'] = df['cost']

        for n in tqdm(range(nb_it)):
            if verbose:
                print("n", n)

            # training an estimate of q_1
            if reg is not None:
                # using previous classifier as estimate of q_n-1
                states = df[["{}_n".format(var) for var in self.state_variables]].copy()
                #display("shape", states.shape, states.dropna().shape, df.shape, df.dropna().shape)
                preds = []
                for a in self.actions:
                    states['action'] = a
                    # display(a, states.shape, states.dropna().shape)
                    preds.append(reg.predict(states))
                preds = pd.DataFrame(preds, index=self.actions, columns=df.index)
                
                # preparing our new training data set
                # if done, do not add any future costs anymore.
                df['min_cost'] = (df['cost'] + (1 - df['done']) * (preds.min() * d_rate))
                
            reg = RandomForestRegressor(n_estimators=20, max_depth=max_depth)
            reg = reg.fit(df[self.state_variables+['action']], df['min_cost'])
            if verbose:
                print("Score:", reg.score(df[self.state_variables+['action']], df['min_cost']))
                print("Feature importances:", reg.feature_importances_)
            self.model = reg

            #self.heatmap_Q(show_traces=False, show_minima_count=True, vol_intervals=10)

    def sample_from_Q(self, vol_intervals, which_min):
        assert len(self.state_variables) == 2, "Not yet implemented for more than 2 variables in state"
        
        df = pd.DataFrame([], columns=self.state_variables)
        for t in range(1, self.T+1):
            for v in np.linspace(0, self.V, num=vol_intervals+1)[1:]:
                
                state = self.generate_state(time_left=t,
                                        volume_left=v)

                q = np.array(self.predict(state))

                action, action_idx = self.get_action(state, exploration=0, which_min=which_min)
                minima_count = len(np.where(q == np.nanmin(q))[0])
                
                # if (t, v) in [(1,100), (3,10), (4,10), (2,10), (1,10)]:
                #     print("t{}, v{}  -  action: #{}={:1.1f}".format(t,v, np.nanargmin(q), action))
                #     print(["{:1.4f}".format(val) for val in q])
                
                df_tmp = pd.DataFrame({'time': t,
                                       'state': str(state),
                                       'volume': v,
                                       'q': np.nanmin(q),
                                       'minima_count': minima_count,
                                       'action': action}, index=["{:1.2f},{}".format(v, t)])
                df = pd.concat([df, df_tmp])

        return df
