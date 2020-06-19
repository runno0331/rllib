import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from models.agent.policygrad import PolicyGrad


class ActorCritic(PolicyGrad):
    def add_memory(self, reward, prob, state_value):
        self.memory.append((reward, prob, state_value))

    def get_action(self, state, greedy=False):
        prob = None
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.float).view(-1, *self.num_state)
        action_prob, state_value = self.pinet(state_tensor)
        action_prob, state_value = action_prob.squeeze(), state_value.squeeze()

        if greedy:
            action = torch.argmax(action_prob).item()
        else:
            action = Categorical(action_prob).sample().item()
            prob = action_prob[action]

        return action, prob, state_value

    def train(self):
        R = 0
        actor_loss = 0
        critic_loss = 0
        for r, prob, v in self.memory[::-1]:
            R = r + self.gamma * R
            advantage = R - v
            actor_loss -= torch.log(prob) * advantage
            critic_loss += F.smooth_l1_loss(v, torch.tensor(R))
        actor_loss = actor_loss / len(self.memory)
        critic_loss = critic_loss / len(self.memory)
        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()

        return loss
