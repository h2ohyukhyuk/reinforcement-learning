# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

env = gym.make("CartPole-v1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
DISCOUNT = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0
episode_durations = []

def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if random.random() < eps_threshold:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model():

    transitions = memory.sample(BATCH_SIZE) # [t1, t2, ..., tn]
    batch = Transition(*zip(*transitions))  # *-> (s,a,n,r)s zip-> [ss,as,ns,rs] *-> ss,as,ns,rs

    non_final_mask = tuple(map(lambda s: bool(s != None), batch.next_state))
    non_final_mask = torch.tensor(non_final_mask, device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state) # batch x n_state
    action_batch = torch.cat(batch.action) # batch x n_action
    reward_batch = torch.cat(batch.reward) # batch

    state_action_values = policy_net(state_batch) # batch x n_action
    state_action_values = state_action_values.gather(1, action_batch) # batch x 1

    next_max_reward_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_reward = target_net(non_final_next_states) # batch x n_actions
        values, indices = next_reward.max(1) # (batch - n_final)
        next_max_reward_values[non_final_mask] = values

    target_action_values = reward_batch + DISCOUNT * next_max_reward_values

    criterion = nn.SmoothL1Loss().to(device)
    loss = criterion(state_action_values, target_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def train():
    '''

    - select action from policy net or random
    - add transition to replay memory continuously
    - train policy net with next_reward of target net
        state -> (policy net, random) -> action -> (env) -> reward, next state
        next state -> (target net) -> next reward
        action value == reward + discount * next_reward
    - soft update target net by policy net

    '''
    path_summary = '../runs/cart-pole/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(path_summary, exist_ok=True)
    sw = SummaryWriter(path_summary)

    num_episodes  = 600 if torch.cuda.is_available() else 50

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = select_action(state)

            observation, reward, terminated, truncated, info = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state

            if len(memory) > BATCH_SIZE:
                optimize_model()

                # Soft update of the target network's weights
                policy_net_state_dict = policy_net.state_dict()
                target_net_state_dict = target_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = 0.995 * target_net_state_dict[key] + 0.005 * policy_net_state_dict[key]
                target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations(show_result=False)
                sw.add_scalar(tag='duration', scalar_value=t+1, global_step=i_episode)
                break

    torch.save(policy_net.state_dict(), '../models/cart-pole.pth')
    sw.close()
    print('Complete')
    plot_durations(show_result=True)
    plt.savefig(path_summary + '/cart-pole-durations.png')

def test_render():

    random_action_prob = 0.05
    env = gym.make("CartPole-v1", render_mode="human")
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    policy_net.load_state_dict(torch.load('../models/cart-pole.pth'))

    for i in range(10000):

        if random.random() < random_action_prob:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state).max(1).indices.view(1, 1).item()

        observation, reward, terminated, truncated, info = env.step(action)

        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        done = terminated or truncated

        if reward != 1:
            break

#train()
test_render()
