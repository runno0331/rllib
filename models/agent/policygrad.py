import torch
from torch.distributions import Categorical


class PolicyGrad:
    def __init__(self, pinet, optimizer, device, gamma=0.99):
        self.pinet = pinet
        self.optimizer = optimizer
        self.device = device
        self.gamma = gamma

        self.num_state = self.pinet.input_size
        self.num_action = self.pinet.output_size
        self.memory = []

    # save model to save_dir
    def save(self, save_dir):
        if save_dir is not None:
            torch.save(self.pinet.state_dict(), save_dir)
            print("Saved model!")

    # load model from load_dir
    def load(self, load_dir):
        if load_dir is not None:
            state_dict = torch.load(load_dir)
            self.pinet.load_state_dict(state_dict)
            print("Loaded model!")
        else:
            raise ValueError("Load Directory")

    def add_memory(self, reward, prob):
        self.memory.append((reward, prob))

    def reset_memory(self):
        self.memory = []

    def get_action(self, state, greedy=False):
        prob = None
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.float).view(-1, *self.num_state)
        action_prob = self.pinet(state_tensor).squeeze()

        if greedy:
            action = torch.argmax(action_prob).item()
        else:
            action = Categorical(action_prob).sample().item()
            prob = action_prob[action]

        return action, prob

    def train(self):
        R = 0
        loss = 0
        for r, prob in self.memory[::-1]:
            R = r + self.gamma * R
            loss -= torch.log(prob) * R
        loss = loss / len(self.memory)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().numpy()
