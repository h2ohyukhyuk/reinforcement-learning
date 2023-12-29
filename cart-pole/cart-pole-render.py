# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human")

num_episode = 600

for i_episode in range(num_episode):
    state, info = env.reset()

    while True:
        env.render()
        action = env.action_space.sample()

        observation, reward, terminated, truncated, _ = env.step(action)

        print(f'observation: {observation} reward: {reward}')

        done = terminated or truncated

        #if done:
        if reward != 1:
            break
    print('-'*50)
    time.sleep(3)