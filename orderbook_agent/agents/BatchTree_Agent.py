from tqdm import tqdm
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor
import random

from .RL_Agent_Base import RLAgent_Base
from helper.orderbook_trader import OrderbookTradingSimulator


class RLAgent_BatchTree(RLAgent_Base):
    def __init__(self, actions, V, T, period_length, model=None, samples=None,
                 agent_name='BatchTree_Agent', state_variables=['volume', 'time'],
                 normalized=True):
        super().__init__(
            actions=actions,
            V=V,
            T=T,
            period_length=period_length,
            agent_name=agent_name,
            state_variables=state_variables,
            normalized=normalized)
        self.model = model
        
        self.columns = state_variables + ['action', 'action_idx', 'cost', 'done'] + [var + "_n" for var in state_variables]
        self.samples = pd.DataFrame(samples, columns=self.columns)

    def predict(self, states):
        df = pd.DataFrame(states).T
        
        preds = []
        for a in self.actions:
            d = pd.concat([df, pd.DataFrame(np.ones(df.shape[0])*a)], axis=1)
            preds.append(self.model.predict(d)[0])

        return preds

    def get_action(self, p_feat, exploration):
        """
        given the q-function estimage (self.model), the state p_feat,
        an exploration noise threshold, and an action space, returns action
        """
        
        if self.model is None or random.random() < exploration:
            action_idx = random.randint(0, len(self.actions)-1)
            action = self.actions[action_idx]
        else:   
            preds = self.predict(p_feat)
            action_idx = np.argmin(preds)
            action = self.actions[action_idx]
            
        return action, int(action_idx)
        
    def learn(self, state, action, cost, new_state):
        print("learn")
        raise NotImplementedError    

    def append_samples(self, state, action, action_idx, cost, done, new_state):
        tmp = pd.DataFrame([state.tolist() + [action] + [action_idx] + [cost] + [done] + new_state.tolist()], columns=self.columns)
        self.samples = pd.concat([self.samples, tmp], axis=0, ignore_index=True)

    def fitted_Q_iteration_tree(self, nb_it):
        reg = None
        d_rate = 0.95
        df = self.samples.copy()
        df['min_cost'] = df['cost']
        # display(df)
        for n in range(nb_it):
            
            # training an estimate of q_1
            if reg is not None:
                # using previous classifier as estimate of q_n-1
                states = df.iloc[:, -self.state_dim-1:-1]

                preds = []
                for a in self.actions:
                    d = pd.concat([states, pd.DataFrame(np.ones(states.shape[0])*a)], axis=1)
                    preds.append(reg.predict(d))
            
                # preparing our new training data set
                # if done, do not add any future costs anymore.
                df['min_cost'] = df['cost'] + (1 - df['done']) * (np.amin(preds, axis=0) * d_rate)
            else:
                df['min_cost'] = df['cost']
            
            features = df.columns[:self.state_dim+1]
            reg = RandomForestRegressor(max_depth=5)
            reg = reg.fit(df[features], df['min_cost'])
            
        self.model = reg
