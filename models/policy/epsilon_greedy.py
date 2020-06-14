import random
import torch
import numpy as np

class EpsilonGreedy:
    def __init__(self, start_eps):
        self.start_eps = start_eps
        self.total_steps = 0
    
    def get_action(self, state, qnet, device, greedy=False):

        epsilon = self.start_eps
        if not greedy:
            self.total_steps +=1

        if greedy or epsilon < random.random():
            with torch.no_grad():
                return qnet(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(qnet.output_size)]], device=device)

class EpsilonGreedyLinearDecay(EpsilonGreedy):
    def __init__(self, start_eps, end_eps, decay_step):
        super().__init__(start_eps)
        self.end_eps = end_eps
        self.decay_step = decay_step

    def get_action(self, state, qnet, greedy=False):
        raise NotImplementedError

class EpsilonGreedyExpDecay(EpsilonGreedy):
    def __init__(self, start_eps, end_eps, decay_step):
        super().__init__(start_eps)
        self.end_eps = end_eps
        self.decay_step = decay_step

    def get_action(self, state, qnet, device, greedy=False):
        epsilon = self.end_eps + (self.start_eps - self.end_eps) * np.exp(-1.0 * self.total_steps / self.decay_step)
        if not greedy:
            self.total_steps +=1

        if greedy or epsilon < random.random():
            with torch.no_grad():
                return qnet(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(qnet.output_size)]], device=device)