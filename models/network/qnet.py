import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, output_size)

        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        h = F.elu(self.lin1(x))
        h = F.elu(self.lin2(h))
        y = F.elu(self.lin3(h))
        return y