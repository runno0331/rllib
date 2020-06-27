import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class PolicyNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=256, action_range=1.0):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state[0], hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_action[0])
        self.fc4 = nn.Linear(hidden_size, num_action[0])

        self.num_state = num_state
        self.num_action = num_action
        self.action_range = action_range

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mean = self.fc3(h)
        log_std = self.fc4(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)

        normal = Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)

        action_greedy = torch.tanh(mean) * self.action_range
        action = y * self.action_range

        log_prob = normal.log_prob(x) - torch.log(self.action_range*(1.0 - torch.pow(y, 2.0)) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action_greedy, action, log_prob


class SoftQNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=256):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state[0]+num_action[0], hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.num_state = num_state
        self.num_action = num_action

    def forward(self, state, action):
        h = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        h = F.relu(self.fc2(h))
        y = self.fc3(h)
        return y
