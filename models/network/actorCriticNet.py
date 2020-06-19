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
    def __init__(self, num_state, num_action, hidden1_size=400, hidden2_size=300, init_w=3e-3):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state[0], hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, num_action[0])

        self.bn1 = nn.BatchNorm1d(hidden1_size)
        self.bn2 = nn.BatchNorm2d(hidden2_size)

        self.num_state = num_state
        self.num_action = num_action

        self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
        self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y = 1.0*torch.tanh(self.fc3(h))
        return y


class CriticNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden1_size=400, hidden2_size=300, init_w=3e-3):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(num_state[0], hidden1_size)
        self.fc2 = nn.Linear(hidden1_size+num_action[0], hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 1)

        self.bn1 = nn.BatchNorm1d(hidden1_size)
        
        self.num_state = num_state
        self.num_action = num_action

        self.fc1.weight.data = init_weight(self.fc1.weight.data.size())
        self.fc2.weight.data = init_weight(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, action):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(torch.cat([h, action], dim=1)))
        y = self.fc3(h)
        return y
