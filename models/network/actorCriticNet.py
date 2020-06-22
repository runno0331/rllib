import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCriticNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(input_size[0], hidden_size)
        self.fc2actor = nn.Linear(hidden_size, output_size)
        self.fc2critic = nn.Linear(hidden_size, 1)
        
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        h = F.elu(self.fc1(x))
        action_prob = F.softmax(self.fc2actor(h), dim=-1)
        state_value = self.fc2critic(h)
        return action_prob, state_value

def init_weight(size):
    f = size[0]
    v = 1. / np.sqrt(f)
    return torch.tensor(np.random.uniform(low=-v, high=v, size=size), dtype=torch.float)


class ActorNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden1_size=400, hidden2_size=300, init_w=3e-3, perturb=False):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state[0], hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, num_action[0])

        if perturb:
            self.bn0 = nn.LayerNorm(num_state[0])
            self.bn1 = nn.LayerNorm(hidden1_size)
            self.bn2 = nn.LayerNorm(hidden2_size)
        else:
            self.bn0 = nn.BatchNorm1d(num_state[0])
            self.bn1 = nn.BatchNorm1d(hidden1_size)
            self.bn2 = nn.BatchNorm1d(hidden2_size)

        self.num_state = num_state
        self.num_action = num_action

        self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
        self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        h = self.bn0(x)
        h = F.relu(self.fc1(h))
        h = self.bn1(h)
        h = F.relu(self.fc2(h))
        h = self.bn2(h)
        y = torch.tanh(self.fc3(h))
        return y


class CriticNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden1_size=400, hidden2_size=300, init_w=3e-4, perturb=False):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state[0], hidden1_size)
        self.fc2 = nn.Linear(hidden1_size+num_action[0], hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 1)

        if perturb:
            self.bn0 = nn.LayerNorm(num_state[0])
            self.bn1 = nn.LayerNorm(hidden1_size)
            self.bn2 = nn.LayerNorm(hidden2_size)
        else:
            self.bn0 = nn.BatchNorm1d(num_state[0])
            self.bn1 = nn.BatchNorm1d(hidden1_size)
        
        self.num_state = num_state
        self.num_action = num_action
        self.perturb = perturb

        self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
        self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, action):
        h = self.bn0(x)
        h = F.relu(self.fc1(h))
        h = self.bn1(h)
        h = F.relu(self.fc2(torch.cat([h, action], dim=1)))
        if self.perturb:
            h = self.bn2(h)
        y = self.fc3(h)
        return y
