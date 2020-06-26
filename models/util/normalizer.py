import numpy as np

class Normalizer:
    def __init__(self, num_states):
        self.n = 0
        self.mean = np.zeros(num_states)
        self.mean_diff = np.zeros(num_states)
        self.var = np.zeros(num_states)

    def update(self, x):
        self.n += 1
        prev_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - prev_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var)
