import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class QLearn:
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        self.q = {}
        self.n = {}  # n is the number of times we have tried an action in a state

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        self.actions_count = len(actions)

    def get_action_index(self, action):
        action_idx = None
        assert action in self.actions, "Action not found: {}".format(action)
        
        action_idx = self.actions.index(action)

        return action_idx

    def getQ(self, state, action=None):

        Q_values = self.q.get(state, np.zeros(self.actions_count))
        
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
            
            self.q[state][action_idx] = n/(n+1) * oldv[action_idx] + 1/(n+1) * (cost + np.min(self.getQ(new_state)))

            
            self.n[state][action_idx] = n + 1

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            print("random")
        else:
            q = self.getQ(state)

            min_indices = np.where(q == q.min())[0]
            
            if len(min_indices) > 1:
                i = random.choice(min_indices)
            else:
                i = min_indices[0]
            
            action = self.actions[i]
        return action

    def plot_Q2(self, V, T, I=5, outfile=None, outformat='pdf'):
        print(V, I)

        volumes = np.linspace(0, V, num=I)[1:]
        timesteps = range(1, T+1)
        grey_tones = np.linspace(1, 0.3, num=T)
        
        print("volumes: {}".format(volumes))
        print("V: {}, T: {}".format(V, T))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        length = (T+1) * I
        
        xpos = []
        ypos = []
        zpos = np.zeros(length)

        dx = np.ones(length) * V/(I-1) / T*2.

        dy = np.ones(length) / 2.
        dz = []
        colors = []
        
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
                    dz.append(self.actions[np.argmin(q)])

        col = ['red', 'blue', 'magenta', 'green', 'yellow', 'grey']
        cmap=plt.get_cmap('Greys')

        ax.bar3d(ypos, xpos, zpos, dx, dy, dz, color=colors, alpha=1)
        
        ax.set_ylabel("t")
        ax.invert_xaxis()
        plt.yticks(timesteps)
        ax.set_xlabel("shares remaining")
        plt.xticks(volumes)
        
        ax.set_zlabel("optimal action")
        ax.set_zlim3d((0,18))

        plt.title("Q function")
        plt.show()

    def plot_Q(self, V, T, I=5, outfile=None, outformat='pdf'):
        # deprecated
        volumes = np.linspace(0, V, num=I)
        print("volumes", volumes)
        print(V, T, 1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for t in range(T+1)[::-1]:
            xs = volumes
            
            ys = np.zeros(I)
            for j, v in enumerate(volumes):
                # print(t, v)
                state = '[{}, {:2d}]'.format(t,int(v))
                q = self.getQ(state)
                # print(state, q, self.actions[np.argmin(q)])
                ys[j] = self.actions[np.argmax(q)]
                # state = np.array([t, v])
                # qval = model.predict(state.reshape(1, STATE_DIM))
                # ys[v] = actions[np.argmin(qval)]
            print("t", t)
            print("ys", ys)
            colors = [['red', 'green'][int(x)] for x in (ys >= 0)]
            

            ax.bar(xs, ys, zs=int(t), zdir='y', color=colors, alpha=0.5)

        ax.set_zlim3d((-3,max(ys)))

        
        ax.invert_xaxis()
        ax.invert_yaxis()
        # ax.invert_zaxis()
        ax.set_xlabel("shares remaining")
        ax.set_ylabel("time remaining")
        ax.set_zlabel("action")
        plt.title("Q function")
        if outfile:
            if outfile[-3:] != outformat:
                outfile = "{}.{}".format(outfile, outformat)
            plt.savefig(outfile, format=outformat)
            print("Successfully saved '{}'".format(outfile))
        else:
            plt.show()
        plt.close()



import math
def ff(f,n):
    fs = "{:f}".format(f)
    if len(fs) < n:
        return ("{:"+n+"s}").format(fs)
    else:
        return fs[:n]