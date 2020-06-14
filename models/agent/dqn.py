import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from models.buffer.buffer import Transition, ReplayBuffer

class DQN:
    def __init__(self, qnet, policy, optimizer, replay_buffer, device, gamma=0.99, batch_size=32, update_target_interval=1):
        self.qnet = qnet
        self.target_qnet = copy.deepcopy(self.qnet)
        self.policy = policy
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_interval = update_target_interval

        self.num_state = qnet.input_size
        self.num_action = qnet.output_size

        self.total_steps = 0

    def save(self, save_dir):
        if save_dir is not None:
            torch.save(self.qnet.state_dict(), save_dir)
            print("Saved model!")

    def load(self, load_dir):
        if load_dir is not None:
            state_dict = torch.load(load_dir)
            self.qnet.load_state_dict(state_dict)
            self.target_qnet.load_state_dict(load_dir)
            print("Loaded model!")
        else:
            raise ValueError("Load Directory")

    def take_action(self, state, greedy=False):
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.float).view(-1, self.num_state)
        action = self.policy.get_action(state_tensor, self.qnet, self.device, greedy)

        return action.item()

    def calculate_next_state_values(self, next_state):
        return self.target_qnet(next_state).max(1)[0].detach()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(batch.state, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.long).unsqueeze(1)
        next_state_batch = torch.tensor(batch.next_state, device=self.device, dtype=torch.float)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float)
        not_done = np.array([(not done) for done in batch.done])
        not_done_batch = torch.tensor(not_done, device=self.device, dtype=torch.float)

        state_action_values = self.qnet(state_batch).gather(1, action_batch)

        next_state_values = self.calculate_next_state_values(next_state_batch)

        expected_state_action_values = reward_batch + (self.gamma * next_state_values * not_done_batch) 

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()

        # print(loss)

        loss.backward()
        for param in self.qnet.parameters():
            # clipping grad
            param.grad.data.clamp_(-1.0, 1.0)
        # print(param.grad.data)

        self.optimizer.step()

        self.total_steps += 1

        if self.total_steps % self.update_target_interval == 0:
            self.target_qnet = copy.deepcopy(self.qnet)

        return loss.detach().numpy()