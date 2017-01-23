from collections import deque  # FIFO queue
import random

from keras.models import Sequential

class Memory:  # stored as (S, A, R, S')
    
    def __init__(self, capacity):
        self.content = deque(maxlen=capacity)
        self.capacity = capacity
        
    def add(self, sample):
        self.content.append(sample)
        
    def size(self):
        return len(self.content)
        
    def get_random_samples(self, num_samples, include_most_recent=True):
        num_samples = min(num_samples, self.size())
        
        if include_most_recent:
            # make sure most recent element is always contained in random selection
            most_recent_element = self.content.pop()
            random_samples = random.sample(self.content, num_samples-1)
            random_samples.append(most_recent_element)
            self.content.append(most_recent_element)
            return random_samples
        
        return random.sample(self.content, num_samples)

    
class Q_learner:  # Reinforcement Learning Agent
    
    MAX_EXPLORATION_RATE = 1.
    MIN_EXPLORATION_RATE = 0.05
    epsilon = 1.  # initial epsilon for greedy action selection
    DECAY_RATE = 0.005   # for shrinkage of epsilon
    LEARNING_RATE = 0.9  # gamma
    
    def __init__(self, model, state_dim, num_actions, n_step):
        assert isinstance(model, Sequential), 'Agent model should be a Sequential Keras model'
        self.model = model
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.n_step = n_step
        
    def choose_action(self, state):
        if random.random() < self.epsilon:
            # choose random action
            action = random.randint(0, self.num_actions-1)
        else:
            # choose best action from Q(s,a) values
            qval = model.predict(state.reshape(1, self.state_dim))
            action = np.argmax(qval)

        return action
        
    def train(self, X_train, y_train, batch_size=None, verbose=0):
        assert(len(X_train) == len(y_train))
        if batch_size is None:
               batch_size = len(X_train)
               
        self.model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=verbose)
    
    def predict(self, state):
        prediction = self.model.predict(state.reshape(1, self.state_dim))
        return prediction