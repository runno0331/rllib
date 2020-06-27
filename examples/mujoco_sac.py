import sys
sys.path.append('../')
import torch
import torch.optim as optim
import gym
import numpy as np
from models.agent.sac import SAC
from models.network.softActorCritic import PolicyNetwork, SoftQNetwork
from models.buffer.buffer import ReplayBuffer
from models.logger.logger import Logger

max_episodes = 1000
memory_capacity = 1e6
gamma = 0.99
alpha = 0.2  # temperature
reward_scale = 5.0  # 20 for humanoid else 5
tau = 5e-3  # soft target update
batch_size = 256
lr = 3e-4
action_range = 1.0

warmup_steps = 1000
logger_interval = 1
render_interval = 20

# env = gym.make('Pendulum-v0')
env = gym.make('HalfCheetah-v2')
# env = gym.make('Humanoid-v2')

num_state = env.observation_space.shape
num_action = env.action_space.shape
max_steps = env.spec.max_episode_steps
hidden_size = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

policyNet = PolicyNetwork(num_state, num_action, hidden_size, action_range)
qnet1 = SoftQNetwork(num_state, num_action, hidden_size)
qnet2 = SoftQNetwork(num_state, num_action, hidden_size)
optimizer_policy = optim.Adam(policyNet.parameters(), lr=lr)
optimizer_qnet1 = optim.Adam(qnet1.parameters(), lr=lr)
optimizer_qnet2 = optim.Adam(qnet2.parameters(), lr=lr)
replay_buffer = ReplayBuffer(capacity=memory_capacity)
agent = SAC(policyNet, qnet1, qnet2, optimizer_policy, optimizer_qnet1, optimizer_qnet2, replay_buffer, device, gamma, alpha, reward_scale, tau, batch_size, action_range)
logger = Logger()

print('Warming up')
step_count = 0
while step_count < warmup_steps:
    observation = env.reset()
    for step in range(max_steps):
        action = env.action_space.sample()
        next_observation, reward, done, _ = env.step(action)
        agent.add_memory(observation, action, next_observation, reward*reward_scale, done)
        step_count += 1
        observation = next_observation

        if done:
            break

step_count = 0
print("Start training")
for episode in range(max_episodes):
    observation = env.reset()
    total_reward = 0
    step_count += 1

    for step in range(max_steps):
        action = agent.get_action(observation)
        next_observation, reward, done, _ = env.step(action)
        step_count += 1
        total_reward += reward
        agent.add_memory(observation, action, next_observation, reward*reward_scale, done)

        agent.train()
        observation = next_observation

        if done:
            break
    
    logger.update(total_reward=total_reward)
    if episode % logger_interval == 0:
        print("episode:{} total reward:{}".format(episode, total_reward))
        print("Current environment steps:{}".format(step_count))

    if episode % render_interval == 0:
        temp = 0
        observation = env.reset()
        env.render()
        for step in range(max_steps):
            action = agent.get_action(observation, greedy=True)
            # print(action)
            next_observation, reward, done, _ = env.step(action)
            agent.add_memory(observation, action, next_observation, reward, done)
            temp += reward
            observation = next_observation
            env.render()

            if done:
                break
        print("test reward:{}".format(temp))

print("Final environment steps:{}".format(step_count))
logger.show_total_reward_record()

for episode in range(3):
    observation = env.reset()
    env.render()
    for step in range(max_steps):
        action = agent.get_action(observation, greedy=True)
        next_observation, _, done, _ = env.step(action)
        observation = next_observation
        env.render()

        if done:
            break

env.close()
