import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import math

class QLearn:
    def __init__(self, actions, epsilon=0.1):
        self.q = {}
        self.n = {}  # n is the number of times we have tried an action in a state

        self.epsilon = epsilon
        self.actions = actions
        self.actions_count = len(actions)

    def __str__(self):
        return("States: {}\nActions: {}".format(self.q, self.actions))

    def save(self, outfile):
        if outfile[-1:] != 'p':
            outfile = '{}.p'.format(outfile)

        with open(outfile, 'wb') as f:
            pickle.dump(self, f)
        print("Saved: '{}'".format(outfile))
            # json.dump([self.q, self.n, self.epsilon, self.actions], f)

    def load(self, infile):
        if infile[-1:] != 'p':
            infile = '{}.p'.format(infile)

        with open(infile, 'rb') as f:
            instance = pickle.load(f)
            return instance
            



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

        if oldv is None:
            q = self.getQ(state)
            
            q[action_idx] = cost
            self.q[state] = q

            init_n = self.getN(state)
            init_n[action_idx] = 1
            self.n[state] = init_n

        else:
            n = self.getN(state, action=action)
            
            if math.isnan(np.nanmin(self.getQ(new_state))):
                minQ_new_state = 0
            else:
                minQ_new_state = np.nanmin(self.getQ(new_state))

            if n == 0:
                self.q[state][action_idx] = (cost + minQ_new_state)
            else:
                self.q[state][action_idx] = n/(n+1) * self.q[state][action_idx] + 1/(n+1) * (cost + minQ_new_state)

            # self.q[state][action_idx] = n/(n+1) * old + 1/(n+1) * (cost + np.nanmin(self.getQ(new_state)))
            
            self.n[state][action_idx] = n + 1

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            print("random")
        else:
            q = self.getQ(state)
            assert np.isnan(q).all() == False, "q table is empty for state '{}'. Probably a missmatch between parameter 'V' used for training and used now.".format(state) 
            
            min_indices = np.where(q == q.min())[0]
            
            if len(min_indices) > 1:
                i = random.choice(min_indices)
            else:
                i = min_indices[0]
            
            action = self.actions[i]
        return action

    def plot_Q(self, V, T, I=5, z_represents='action', outfile=None, outformat='pdf'):
        assert isinstance(z_represents, str) and z_represents in ['action', 'Q', 'both']
        
        volumes = np.linspace(0, V, num=I)[1:]
        timesteps = range(1, T+1)
        grey_tones = np.linspace(1, 0.3, num=T)
        
        fig = plt.figure()
        ax = [fig.add_subplot(111, projection='3d')]
        
        length = T * (I-1)
        
        xpos = []
        ypos = []
        zpos = np.zeros(length)

        dx = np.ones(length) * V/(I-1) / T  * 4.

        dy = np.ones(length)  # / 2.
        colors = []

        # dz:
        dz = []
        dz_action = []
        dz_q = []
        
        count = 0
        for t in timesteps:
            xs = volumes

            ys = np.zeros(I)
            for j, v in enumerate(volumes):
                state = '[{}, {:2d}]'.format(t,int(v))
                
                xpos.append(t)
                ypos.append(int(v))
                colors.append((grey_tones[t-1], grey_tones[t-1], grey_tones[t-1]))
                
                q = self.getQ(state)
                

                if all(x==q[0] for x in q):
                    # same q value everywhere
                    dz.append(0)
                else:
                    if np.all([math.isnan(val) for val in q]):
                        xpos.pop()
                        ypos.pop()
                        colors.pop()
                        dx = dx[:-1]
                        dy = dy[:-1]
                    else:
                        dz_action.append(self.actions[np.nanargmin(q)])
                        dz_q.append(np.nanmin(q))

        col = ['red', 'blue', 'magenta', 'green', 'yellow', 'grey']
        cmap=plt.get_cmap('Greys')

        if z_represents == 'action':
            ax[0].bar3d(ypos, xpos, zpos, dx, dy, dz=dz_action, color=colors, alpha=1)
            ax[0].set_zlabel("optimal action")
        elif z_represents == 'Q':
            ax[0].bar3d(ypos, xpos, zpos, dx, dy, dz=dz_q, color=colors, alpha=1)
            ax[0].set_zlabel("Q Value")
        elif z_represents == 'both':
            plt.close()
            fig = plt.figure(figsize=(10,3))
            ax = [fig.add_subplot(111, projection='3d')]
            ax[0].bar3d(ypos, xpos, zpos, dx, dy, dz=dz_action, color=colors, alpha=1)

            ax.append(fig.add_subplot(122, projection='3d'))
            ax[1].bar3d(ypos, xpos, zpos, dx, dy, dz=dz_q, color=colors, alpha=1)
            ax[1].set_zlabel("Q Value")



        # layout
        for axis in ax:
            axis.set_ylabel("t")
            axis.set_xlim3d((volumes[0],volumes[-1]+volumes[0]))
            axis.invert_xaxis()
            
            axis.set_ylim3d((1,T+1))
            axis.set_yticks(timesteps)
            
            axis.set_xlabel("shares remaining")
            axis.set_xticks(volumes)
            plt.tight_layout()
            # axis.set_zlim3d((0,18))

        

        fig.suptitle("Q function")
        
        if outfile:
            if outfile[len(outformat):] != outformat:
                outfile = "{}.{}".format(outfile, outformat)
            plt.savefig(outfile, format=outformat)
            print("Successfully saved '{}'".format(outfile))
        else:
            plt.show()
        plt.close()
