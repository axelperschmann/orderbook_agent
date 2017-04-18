from collections import deque  # FIFO queue
import random

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
