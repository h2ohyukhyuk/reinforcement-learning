# references
# https://gymnasium.farama.org/environments/toy_text/frozen_lake/
# https://www.youtube.com/watch?v=yOBKtGU6CG0&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG&index=5
 # Q is a policy function given state and action or only state
 # pi: policy, *: optimal
 # pi*(state) = argmax_a Q(state, action)
 # Q^ <- r + discount_factor * max Q(new_state, actions)
 # gamma: discounted future reward factor
 # Q^ converges to Q in deterministic worlds and in finite states, machine learning, tom mitchell, 1997

import gym
import numpy as np
import matplotlib.pyplot as plt

import random

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

print('gym version: ', gym.__version__) # 0.26.2

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)

desc=["SFFF",
      "FHFH",
      "FFFH",
      "HFFG"]
env = gym.make('FrozenLake-v1', desc=desc, map_name="4x4", is_slippery=True)

Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000
discount_factor = 0.9
lr = 0.25

list_reward = []

def exploit_exploration_decay_e_greedy():
    for i in range(num_episodes):
        state, info = env.reset()
        all_reward = 0
        done = False

        while not done:
            # exploit_exploration E&E
            if random.random() < (1 - (i + 1) / num_episodes)*0.1:
                action = env.action_space.sample()
            else:
                action = rargmax(Q[state, :])

            # terminated: the player moves into a hole or the player reaches the goal
            # truncated: the length of the episode is 100 for 4x4 environment.
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if state == new_state:
                continue

            if terminated and reward <= 0:
                Q[state, action] = -1
                continue

            future_reward = np.max(Q[new_state,:])
            Q[state, action] = (1 - lr) * Q[state, action] + lr * (reward + discount_factor * future_reward)

            all_reward += reward
            state = new_state
        list_reward.append(all_reward)


def exploit_exploration_add_noise():
    for i in range(num_episodes):
        state, info = env.reset()
        all_reward = 0
        done = False

        while not done:
            # exploit_exploration E&E
            random_noise = np.random.rand(env.action_space.n) * (1 - (i + 1) / num_episodes)
            action = rargmax(Q[state, :] + random_noise)

            # terminated: the player moves into a hole or the player reaches the goal
            # truncated: the length of the episode is 100 for 4x4 environment.
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if state == new_state:
                continue

            if terminated and reward <= 0:
                Q[state, action] = -1
                continue

            future_reward = np.max(Q[new_state, :])
            Q[state, action] = (1-lr)*Q[state, action] + lr*(reward + discount_factor * future_reward)

            all_reward += reward
            state = new_state
        list_reward.append(all_reward)

exploit_exploration_decay_e_greedy()
#exploit_exploration_add_noise()

print(f'Success rate: {sum(list_reward)}/{num_episodes}={sum(list_reward)/num_episodes}')
print('Final Q-Table Values')
print('LEFT DOWN RIGHT UP')

print(Q)
plt.bar(range(len(list_reward)), list_reward, color='b')
plt.show()