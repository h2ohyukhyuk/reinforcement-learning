
import readchar

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
env = gym.make('FrozenLake-v1', desc=desc, map_name="4x4", is_slippery=False, render_mode="human")

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
arrow_keys = {
    'w': UP,
    's': DOWN,
    'd': RIGHT,
    'a': LEFT
}

env.reset()
while True:
    env.render()
    key = readchar.readkey().lower()

    if key == 'q':
        print('Game aborted!')

    if key in arrow_keys:
        new_state, reward, terminated, truncated, info = env.step(arrow_keys[key])

        if truncated:
            print('Exceed the number of move')
            break

        if terminated:
            print(f'Finished with reward {reward}')