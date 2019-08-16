import os
import time
from collections import deque
import json

from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt

from agents import AgentDDPG
from misc import ddpg, find_state_mag, get_env_info, run


env = UnityEnvironment(
    file_name="./Tennis_Linux/Tennis.x86_64", no_graphics=True)

# # find the magnitudes of the different state features
# states_mean, states_std = find_state_mag(env, max_steps=5000, n_episodes=10)
# fig, ax = plt.subplots()
# ax.bar(np.arange(len(states_mean)), states_mean, yerr=states_std)
# ax.set_xlabel("State features")
# fig.savefig("States magnitude.png")

# get info on the environment
brain_name, n_agents, state_size, action_size = get_env_info(env)
print('Number of agents:', n_agents)
print('Size of each action:', action_size)
print('There are {} agents. Each observes a state with length: {}'.format(
    n_agents, state_size))

# Set all parameters
# weight_init can be "uniform" or "kaiming"
# type can be "DDPG", "2 DDPG"
params = {
    "buffer_size": int(1e6),
    "batch_size": 128,
    "gamma": 0.99,
    "tau": 1e-3,
    "lr_actor": 1e-4,
    "lr_critic": 1e-4,
    "weight_decay": 0,
    "update_every": 20,
    "update_steps": 10,
    "double": True,
    "fc1": 400,
    "fc2": 300,
    "batchnorm": True,
    "weight_init": "uniform",
    "seed": 2,
    "scores_window": 100,
    "progress_every": 2,
    "save_every": 60,
    "n_episodes": int(1e5),
    "max_steps": int(5000),
    "verbose": True,
    "max_time": 200*60,
    "score_solved": 0.5,
    "stop_on_solve": True,
    "folder": "trained/test",
    "overwrite": False,
    "type": "DDPG"
}

run(env, params)

env.close()
