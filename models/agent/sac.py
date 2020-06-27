import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from models.buffer.buffer import Transition


class SAC:
    def __init__(self, policy, qnet1, qnet2, optimizer_policy, optimizer_qnet1, optimizer_qnet2, replay_buffer, device, gamma=0.99, alpha=1.0, reward_scale=1.0, tau=5e-3, batch_size=64, action_range=1.0):
        self.policy = policy
        self.qnet1 = qnet1
        self.qnet2 = qnet2
        self.qnet1_target = copy.deepcopy(self.qnet1)
        self.qnet2_target = copy.deepcopy(self.qnet2)
        self.optimizer_policy = optimizer_policy
        self.optimizer_qnet1 = optimizer_qnet1
        self.optimizer_qnet2 = optimizer_qnet2
        self.replay_buffer = replay_buffer
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.reward_scale = reward_scale
        self.tau = tau
        self.batch_size = batch_size
        self.action_range = action_range

        self.log_alpha = torch.zeros(1, dtype=torch.float, requires_grad=True, device=device)
        self.optimizer_log_alpha = optim.Adam([self.log_alpha], lr=3e-4)

        self.num_state = self.policy.num_state
        self.num_action = self.policy.num_action
        self.target_entropy = -self.num_action[0]

    def add_memory(self, *args):
        self.replay_buffer.append(*args)

    def reset_memory(self):
        self.replay_buffer.reset()

    def get_action(self, state, greedy=False, epsilon=1e-6):
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).view(-1, *self.num_state)
        if greedy:
            action, _, _ = self.policy.evaluate(state_tensor)
        else:
            _, action, _ = self.policy.evaluate(state_tensor)
        return action.squeeze(0).detach().cpu().numpy()

    def calculate_next_state_value(self, next_state):
        with torch.no_grad():
            _, next_action, log_prob = self.policy.evaluate(next_state)
            next_qvalue1 = self.qnet1_target(next_state, next_action)
            next_qvalue2 = self.qnet2_target(next_state, next_action)
            next_state_value = torch.min(next_qvalue1, next_qvalue2) - self.alpha * log_prob
        return next_state_value

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(batch.state, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(batch.next_state, device=self.device, dtype=torch.float)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float).unsqueeze(1)
        not_done = np.array([(not done) for done in batch.done])
        not_done_batch = torch.tensor(not_done, device=self.device, dtype=torch.float).unsqueeze(1)

        qvalue1 = self.qnet1(state_batch, action_batch)
        qvalue2 = self.qnet2(state_batch, action_batch)
        next_state_value = self.calculate_next_state_value(next_state_batch)
        qvalue_target = reward_batch + self.gamma * next_state_value * not_done_batch
        q1_loss = 0.5*F.mse_loss(qvalue1, qvalue_target)
        q2_loss = 0.5*F.mse_loss(qvalue2, qvalue_target)
        self.optimizer_qnet1.zero_grad()
        q1_loss.backward()
        self.optimizer_qnet1.step()
        self.optimizer_qnet2.zero_grad()
        q2_loss.backward()
        self.optimizer_qnet2.step()

        _, action, log_prob = self.policy.evaluate(state_batch)
        qvalue1_new = self.qnet1(state_batch, action)
        qvalue2_new = self.qnet2(state_batch, action)
        qvalue_new = torch.min(qvalue1_new, qvalue2_new)
        actor_loss = (self.alpha * log_prob - qvalue_new).mean()
        self.optimizer_policy.zero_grad()
        actor_loss.backward()
        self.optimizer_policy.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.optimizer_log_alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer_log_alpha.step()
        self.alpha = self.log_alpha.exp()

        self.soft_update(self.qnet1_target, self.qnet1)
        self.soft_update(self.qnet2_target, self.qnet2)
