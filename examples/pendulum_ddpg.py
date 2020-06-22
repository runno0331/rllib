import sys
sys.path.append('../')
import torch
import torch.optim as optim
import gym
from models.agent.ddpg import DDPG
from models.network.actorCriticNet import ActorNetwork, CriticNetwork
from models.buffer.buffer import ReplayBuffer
from models.logger.logger import Logger


max_episodes = 10000
memory_capacity = 1e6
gamma = 0.99
tau = 1e-3  # soft target update parameter
epsilon = 1.0
batch_size = 128
lr_actor = 1e-4
lr_critic = 1e-3
weight_decay = 1e-2
logger_interval = 1
render_interval = 200
warmup_steps = 1000
param_noise = False
num_train = 50
sample_step = 100

env = gym.make('HalfCheetah-v2')
# env = gym.make('Hopper-v2')
# env = gym.make('Pendulum-v0')
# env = gym.make('Humanoid-v2')
num_state = env.observation_space.shape
num_action = env.action_space.shape
max_steps = env.spec.max_episode_steps
hidden1_size_actor = 64
hidden2_size_actor = 64
hidden1_size_critic = 64
hidden2_size_critic = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

actorNet = ActorNetwork(num_state, num_action, hidden1_size_actor, hidden2_size_actor, perturb=param_noise).to(device)
criticNet = CriticNetwork(num_state, num_action, hidden1_size_critic, hidden2_size_critic, perturb=param_noise).to(device)
optimizer_actor = optim.Adam(actorNet.parameters(), lr=lr_actor)
optimizer_critic = optim.Adam(criticNet.parameters(), lr=lr_critic, weight_decay=weight_decay)
replay_buffer = ReplayBuffer(capacity=memory_capacity)
agent = DDPG(actorNet, criticNet, optimizer_actor, optimizer_critic, replay_buffer, device, param_noise, gamma, tau, epsilon, batch_size)
logger = Logger()

print('Warming up')
step_count = 0
while step_count < warmup_steps:
    observation = env.reset()
    for step in range(max_steps):
        action = env.action_space.sample()
        next_observation, reward, done, _ = env.step(action)
        agent.add_memory(observation, action, next_observation, reward, done)
        step_count += 1
        observation = next_observation

        if done:
            break

print('Start training')
for episode in range(max_episodes):
    observation = env.reset()
    total_reward = 0
    if not param_noise:
        agent.random_process.reset()

    for step in range(sample_step):
        action = agent.get_action(observation)
        next_observation, reward, done, _ = env.step(action)
        total_reward += reward
        agent.add_memory(observation, action, next_observation, reward, done)

        # agent.train()
        observation = next_observation

        if done:
            break
    
    for _ in range(num_train):
        agent.train()
    if param_noise:
        agent.update_perturbed_actor()

    if episode % render_interval == 0:
        temp = 0
        observation = env.reset()
        env.render()
        for step in range(max_steps):
            action = agent.get_action(observation, greedy=True)
            next_observation, reward, done, _ = env.step(action)
            temp += reward
            observation = next_observation
            env.render()

            if done:
                break
        print("test reward:{}".format(temp))

    if episode % logger_interval == 0:
        print("episode:{} total reward:{}".format(episode, total_reward))
    logger.update(total_reward=total_reward)

logger.show_total_reward_record()

torch.save(agent.actor.state_dict(), './ddpg_actor2')
torch.save(agent.critic.state_dict(), './ddpg_critic2')

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
