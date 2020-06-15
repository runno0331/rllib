import torch.nn as nn
import torch.nn.functional as F


# for discrete action space with non-image input
class PolicyNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size[0], hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc1(x))
        action_prob = F.softmax(self.fc3(h), dim=-1)
        return action_prob
