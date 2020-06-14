from models.agent.dqn import DQN


class DDQN(DQN):
    def calculate_next_state_values(self, next_state):
        next_action_batch = self.qnet(next_state).max(1)[1].unsqueeze(1)
        return self.target_qnet(next_state).gather(1, next_action_batch).squeeze().detach()