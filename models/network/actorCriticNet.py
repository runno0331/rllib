import torch.nn as nn
import torch.nn.functional as F


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
