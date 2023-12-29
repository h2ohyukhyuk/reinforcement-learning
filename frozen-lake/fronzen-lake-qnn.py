'''
Q^(s,a|t) ~ Q*(s,a)
'''

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

print('gym version: ', gym.__version__) # 0.26.2

desc=["SFFF",
      "FHFH",
      "FFFH",
      "HFFG"]
env = gym.make('FrozenLake-v1', desc=desc, map_name="4x4", is_slippery=False)

num_episodes = 2000
discount_factor = 0.99
lr = 0.1

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(env.observation_space.n, 64, bias=False)
        self.layer2 = nn.Linear(64, env.action_space.n, bias=False)

        nn.init.uniform(self.layer1.weight.data, a=-0.01, b=0.01)
        nn.init.uniform(self.layer2.weight.data, a=-0.01, b=0.01)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net()
model.to(device)
model.train()

opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
list_reward = []

for i in range(num_episodes):
    state, info = env.reset()

    all_reward = 0

    for x in range(100):

        state_one_hot = torch.zeros(size=(1, env.observation_space.n))
        state_one_hot[0][state] = 1.0
        state_one_hot = state_one_hot.to(device)

        model.eval()
        with torch.no_grad():
            # action = rargmax(Q[state, :])
            out = model(state_one_hot)
            top_v, top_i = torch.topk(out, k=1, dim=1)
            action = top_i.squeeze().item()

        if random.random() < (1 - (i + 1) / num_episodes):
            action = env.action_space.sample()

        new_state, reward, terminated, truncated, info = env.step(action)

        new_state_one_hot = torch.zeros(size=(1, env.observation_space.n))
        new_state_one_hot[0][new_state] = 1.0
        new_state_one_hot = new_state_one_hot.to(device)
        new_out = model(new_state_one_hot)
        maxQ1 = torch.max(new_out)

        model.train()
        out = model(state_one_hot)
        target = out.clone().detach()

        target[0][action] = reward + discount_factor * maxQ1

        loss = F.mse_loss(target, out)
        if reward == 1:
            print('goal episode: ', i)
            print('state: ', state, 'action: ', action)
            print('target: ', target[0].detach().cpu().numpy())
            print('out: ', out[0].detach().cpu().numpy())
            print(f'loss: {loss.item()}')


        opt.zero_grad()
        loss.backward()
        opt.step()

        all_reward += reward
        state = new_state

        if terminated:
            break

    list_reward.append(all_reward)


print(f'Success rate: {sum(list_reward)}/{num_episodes}={sum(list_reward)/num_episodes}')

#print(Q)
plt.bar(range(len(list_reward)), list_reward, color='b')
plt.show()


