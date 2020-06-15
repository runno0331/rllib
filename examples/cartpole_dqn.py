import sys
sys.path.append('../')
import torch
import torch.optim as optim
from models.buffer.buffer import ReplayBuffer
from models.network.qnet import QNet
from models.agent.dqn import DQN
from models.agent.ddqn import DDQN
from models.policy.epsilon_greedy import EpsilonGreedy, EpsilonGreedyExpDecay
from models.logger.logger import Logger
import numpy as np
import gym

max_episodes = 300
max_steps = 200
capacity = 50000
gamma = 0.99
batch_size = 64
learning_rate = 5e-4
start_eps = 0.9
end_eps = 0.1
decay_step = 5000
ddqn_flag = False

env = gym.make('CartPole-v0')
input_size = env.observation_space.shape
output_size = env.action_space.n
hidden_size = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mainQNet = QNet(input_size, output_size, hidden_size).to(device)
optimizer = optim.Adam(mainQNet.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(capacity=capacity)
policy = EpsilonGreedyExpDecay(start_eps, end_eps, decay_step)
if ddqn_flag:
    network = DDQN(qnet=mainQNet, policy=policy, optimizer=optimizer, replay_buffer=replay_buffer, device=device, gamma=gamma, batch_size=batch_size, update_target_interval=10)
else:
    network = DQN(qnet=mainQNet, policy=policy, optimizer=optimizer, replay_buffer=replay_buffer, device=device, gamma=gamma, batch_size=batch_size)
logger = Logger()

for episode in range(max_episodes):
    observation = env.reset()
    total_reward = 0
    total_loss = []

    for step in range(max_steps):
        action = network.get_action(observation)
        next_observation, reward, done, _ = env.step(action)
        total_reward += reward

        if done and step < 200:
            reward = -1.0
        elif not done:
            reward = 0.0

        network.add_memory(observation, action, next_observation, reward, done)

        loss = network.train()
        if loss is not None:
            total_loss.append(loss)
            # print(loss)

        observation = next_observation

        if done:
            break

    if episode % 10 == 0:
        print("episode:{} total reward:{}".format(episode, total_reward))

    # save in log
    if len(total_loss) == 0:
        total_loss = 0
    else:
        total_loss = np.mean(total_loss)
    logger.update(loss=total_loss, total_reward=total_reward)

logger.show_loss_record()
logger.show_total_reward_record()