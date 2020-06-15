import sys
sys.path.append('../')
import torch
import torch.optim as optim
import gym
from models.network.policyNet import PolicyNet
from models.agent.policygrad import PolicyGrad
from models.logger.logger import Logger


max_episodes = 10000
max_steps = 200
gamma = 0.99
learning_rate = 1e-3
log_interval = 100

env = gym.make('CartPole-v0')
input_size = env.observation_space.shape
output_size = env.action_space.n
hidden_size = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

policyNet = PolicyNet(input_size, output_size, hidden_size).to(device)
optimizer = optim.Adam(policyNet.parameters(), lr=learning_rate)
agent = PolicyGrad(policyNet, optimizer, device, gamma)
logger = Logger()

for episode in range(max_episodes):
    observation = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action, prob = agent.get_action(observation)
        next_observation, reward, done, _ = env.step(action)
        total_reward += reward
        agent.add_memory(reward, prob)
        observation = next_observation
        if done:
            loss = agent.train()
            agent.reset_memory()
            break

    if episode % log_interval == 0:
        logger.update(loss=loss, total_reward=total_reward)
        print("episode:{} total reward:{}".format(episode, total_reward))

logger.show_loss_record()
logger.show_total_reward_record()