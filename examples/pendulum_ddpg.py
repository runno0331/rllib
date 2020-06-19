import sys
sys.path.append('../')
import torch
import torch.optim as optim
import gym
from models.agent.ddpg import DDPG
from models.network.actorCriticNet import ActorNetwork, CriticNetwork
from models.buffer.buffer import ReplayBuffer
from models.logger.logger import Logger

max_episodes = 300
memory_capacity = 1e6
gamma = 0.99
tau = 1e-3  # soft target update parameter
epsilon = 1.0
batch_size = 64
lr_actor = 1e-4
lr_critic = 1e-3
logger_interval = 10

# env = gym.make('HalfCheetah-v2')
env = gym.make('Pendulum-v0')
num_state = env.observation_space.shape
num_action = env.action_space.shape
max_steps = env.spec.max_episode_steps
hidden1_size_actor = 400
hidden2_size_actor = 300
hidden1_size_critic = 400
hidden2_size_critic = 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

actorNet = ActorNetwork(num_state, num_action, hidden1_size_actor, hidden2_size_actor).to(device)
criticNet = CriticNetwork(num_state, num_action, hidden1_size_critic, hidden2_size_critic).to(device)
optimizer_actor = optim.Adam(actorNet.parameters(), lr=lr_actor)
optimizer_critic = optim.Adam(criticNet.parameters(), lr=lr_critic)
replay_buffer = ReplayBuffer(capacity=memory_capacity)
agent = DDPG(actorNet, criticNet, optimizer_actor, optimizer_critic, replay_buffer, device, gamma, tau, epsilon, batch_size)
logger = Logger()

for episode in range(max_episodes):
    observation = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = agent.get_action(observation)
        next_observation, reward, done, _ = env.step(action)
        total_reward += reward
        agent.add_memory(observation, action, next_observation, reward, done)

        agent.train()

        observation = next_observation

        if done:
            break

    if episode % logger_interval == 0:
        print("episode:{} total reward:{}".format(episode, total_reward))
    logger.update(total_reward=total_reward)

logger.show_total_reward_record()

for episode in range(3):
    observation = env.reset()
    env.render()
    for step in range(max_steps):
        action = agent.get_action(observation, greedy=True)
        next_observation, reward, done, _ = env.step(action)
        observation = next_observation
        env.render()

        if done:
            break

env.close()