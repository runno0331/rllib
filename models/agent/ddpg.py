import torch
import torch.nn.functional as F
import numpy as np
import copy
from models.buffer.buffer import Transition
from models.util.noise import OrnsteinUhlenbeckProcess, AdaptiveParamNoise


class DDPG:
    def __init__(self, actor, critic, optimizer_actor, optimizer_critic, replay_buffer, device, param_noise=False, gamma=0.99, tau=1e-3, epsilon=1.0, batch_size=64):
        self.actor = actor
        self.critic = critic
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.replay_buffer = replay_buffer
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.actor_perturb = copy.deepcopy(self.actor)
        self.actor_noise = None
        self.random_process = None
        if param_noise:
            self.actor_noise = AdaptiveParamNoise()
        else:
            self.random_process = OrnsteinUhlenbeckProcess(size=actor.num_action[0])

        self.num_state = actor.num_state
        self.num_action = actor.num_action

    def add_memory(self, *args):
        self.replay_buffer.append(*args)
    
    def reset_memory(self):
        self.replay_buffer.reset()

    def update_perturbed_actor(self):
        if len(self.replay_buffer) > self.batch_size:
            transitions = self.replay_buffer.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.tensor(batch.state, device=self.device, dtype=torch.float)
            actions = self.actor(state_batch)
            actions_perturb = self.actor_perturb(state_batch)
            distance = torch.sqrt(torch.mean(torch.pow(actions - actions_perturb, exponent=2.0))).item()
            self.actor_noise.adapt(distance)

        std = self.actor_noise.get_std()
        for perturb_param, param in zip(self.actor_perturb.fc1.parameters(), self.actor.fc1.parameters()):
            perturb_param.data.copy_(param.data + std*torch.randn_like(param.data))
        for perturb_param, param in zip(self.actor_perturb.fc2.parameters(), self.actor.fc2.parameters()):
            perturb_param.data.copy_(param.data + std*torch.randn_like(param.data))
        for perturb_param, param in zip(self.actor_perturb.fc3.parameters(), self.actor.fc3.parameters()):
            perturb_param.data.copy_(param.data + std*torch.randn_like(param.data))

    def get_action(self, state, greedy=False):
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).view(-1, *self.num_state)

        if not greedy and self.actor_noise is not None:
            self.actor_perturb.eval()
            action = self.actor_perturb(state_tensor)
        else:
            self.actor.eval()
            action = self.actor(state_tensor)
            self.actor.train()

        if not greedy and self.random_process is not None:
            action += self.epsilon*torch.tensor(self.random_process.sample(), dtype=torch.float, device=self.device)
            action = action.clamp(min=-1.0, max=1.0)

        return action.squeeze(0).detach().cpu().numpy()

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

        # need to change
        qvalue = self.critic(state_batch, action_batch)
        next_qvalue = self.critic_target(next_state_batch, self.actor_target(next_state_batch))
        target_qvalue = reward_batch + (self.gamma * next_qvalue * not_done_batch) 

        critic_loss = F.mse_loss(qvalue, target_qvalue)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # soft parameter update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)