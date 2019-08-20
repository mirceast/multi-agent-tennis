import os
import numpy as np
import time
from collections import deque
import glob
import pickle
import shutil
from copy import deepcopy

import matplotlib.pyplot as plt
import torch

from agents import AgentDDPG, AgentMADDPG
from utilities import get_env_info


def run(env, params):
    brain_name, n_agents, state_size, action_size = get_env_info(env)
    if params["type"].lower() == "ddpg":
        agent = AgentDDPG(state_size=state_size,
                          action_size=action_size, params=params)
        scores = ddpg(agent, env, params)
    elif params["type"].lower() == "2 ddpg":
        agent = [AgentDDPG(state_size=state_size, action_size=action_size,
                           params=params) for i in range(n_agents)]
        scores = ddpg(agent, env, params)
    elif params["type"].lower() == "maddpg":
        agent = AgentMADDPG(env, params)
        scores = ddpg(agent, env, params)
    else:
        raise Exception("'type' can be 'ddpg', '2 ddpg', 'maddpg'")


def ddpg(agent, env, params):
    # Get environment information
    brain_name, n_agents, state_size, action_size = get_env_info(env)

    # Initialize stuff
    log = Logger(params, agent)

    for _ in range(1, params["n_episodes"]+1):
        env_info = env.reset(train_mode=True)[brain_name]
        if isinstance(agent, list):
            for i in range(n_agents):
                agent[i].reset()
        else:
            agent.reset()
        states = env_info.vector_observations

        episode_scores = np.zeros(n_agents)
        for t in range(params["max_steps"]):
            if isinstance(agent, list):
                actions = np.zeros((n_agents, action_size))
                for i in range(n_agents):
                    actions[i] = agent[i].act(states[i])
            else:
                actions = agent.act(states)
            if params["type"].lower() == "maddpg":
                actions = actions.reshape(n_agents, action_size)
            actions = actions.detach().cpu().numpy()
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            if isinstance(agent, list):
                for i in range(n_agents):
                    agent[i].step(states[i], actions[i], rewards[i],
                                  next_states[i], dones[i])
            else:
                agent.step(states, actions, rewards, next_states, dones)
            episode_scores += rewards
            states = next_states
            # check if we should save and show progress
            log.tic()
            if np.any(dones):
                break

        log.update(agent, episode_scores, t+1)
        log.tic()
        if log.solved and params["stop_on_solve"]:
            break
        if time.time() - log.t_start > params["max_time"] + 5:
            break
    return agent, log


class Logger():
    '''
    Logs, displays, and saves progress.
    '''

    def __init__(self, params, agent):
        self.data = params

        # If save folder exists raise an exception
        if os.path.isdir(self.data["folder"]):
            if self.data["overwrite"]:
                shutil.rmtree(self.data["folder"])
            else:
                raise Exception("Folder already exists and overwrite is off.")
        if not os.path.isdir(self.data["folder"]):
            os.makedirs(self.data["folder"])

        self.data["scores"] = []
        self.data["mean_scores"] = []
        self.data["steps_done"] = []
        self._update_agent(agent)

        # comb_score_window is the combined score - for tennis it's the max
        self.comb_score_window = deque(maxlen=params["scores_window"])
        # all_score_window contains the scores of all agents
        self.all_score_window = deque(maxlen=params["scores_window"])
        self.best_score = -np.inf
        self.t_start = time.time()
        self.progress_t = time.time()
        self.saved_t = time.time()
        self.solved = False

        self.data["train_time"] = time.time() - self.t_start

    def _update_agent(self, agent):
        if isinstance(agent, list):
            if not "actor_local_dict" in self.data:
                temp = []
                for i in range(len(agent)):
                    temp.append([])
                self.data["actor_local_dict"] = deepcopy(temp)
                self.data["actor_target_dict"] = deepcopy(temp)
                self.data["critic_local_dict"] = deepcopy(temp)
                self.data["critic_target_dict"] = deepcopy(temp)
            else:
                for i in range(len(agent)):
                    self.data["actor_local_dict"][i] = agent[i].actor_local.state_dict()
                    self.data["actor_target_dict"][i] = agent[i].actor_target.state_dict()
                    self.data["critic_local_dict"][i] = agent[i].critic_local.state_dict()
                    self.data["critic_target_dict"][i] = agent[i].critic_target.state_dict(
                    )

        elif isinstance(agent, AgentDDPG):
            self.data["actor_local_dict"] = agent.actor_local.state_dict()
            self.data["actor_target_dict"] = agent.actor_target.state_dict()
            self.data["critic_local_dict"] = agent.critic_local.state_dict()
            self.data["critic_target_dict"] = agent.critic_target.state_dict()

        elif isinstance(agent, AgentMADDPG):
            if not "actor_local_dict" in self.data:
                temp = []
                for i in range(len(agent.maddpg_agent)):
                    temp.append([])
                self.data["actor_local_dict"] = deepcopy(temp)
                self.data["actor_target_dict"] = deepcopy(temp)
                self.data["critic_local_dict"] = deepcopy(temp)
                self.data["critic_target_dict"] = deepcopy(temp)
            else:
                for i in range(len(agent.maddpg_agent)):
                    self.data["actor_local_dict"][i] = agent.maddpg_agent[i].actor_local.state_dict(
                    )
                    self.data["actor_target_dict"][i] = agent.maddpg_agent[i].actor_target.state_dict(
                    )
                    self.data["critic_local_dict"][i] = agent.maddpg_agent[i].critic_local.state_dict(
                    )
                    self.data["critic_target_dict"][i] = agent.maddpg_agent[i].critic_target.state_dict(
                    )

        else:
            raise Exception("Unkown agent type.")

    def update(self, agent, episode_scores, steps):
        self.comb_score_window.append(np.max(episode_scores))
        self.all_score_window.append(episode_scores)

        self.data["scores"].append(episode_scores)
        self.data["mean_scores"].append(np.mean(self.all_score_window, axis=0))
        self.data["steps_done"].append(steps)
        self._update_agent(agent)

        self.tic()

    def show_progress(self):
        if len(self.data["mean_scores"]):
            print('\rMin agent score: {:.2f}\tMax agent score: {:.2f}\tMax steps: {}\tTotal time: {}\tEpisodes: {}'.format(
                min(self.data["mean_scores"][-1]),
                max(self.data["mean_scores"][-1]),
                self.data["steps_done"][-1],
                seconds_to_time_str(time.time() - self.t_start),
                len(self.data["scores"])), end="")
        if len(self.data["mean_scores"]) and self.data["steps_done"][-1] > 5000:
            raise Exception("debug")

    def tic(self):
        self.data["train_time"] = time.time() - self.t_start
        if self.data["verbose"] and (self.data["progress_every"] > 0 and
                                     time.time() - self.progress_t >= self.data["progress_every"]):
            self.show_progress()
            self.progress_t = time.time()

        if self.data["save_every"] > 0 and \
                time.time() - self.saved_t >= self.data["save_every"]:
            self.saved_t = time.time()
            self.save()

        if len(self.comb_score_window) and \
                (np.mean(self.comb_score_window) >= self.data["score_solved"]):
            print('\nEnvironment solved in {:d} episodes!\tAverage combined score: {:.2f}'.format(
                len(self.data["scores"])-100, np.mean(self.comb_score_window)))
            self.save(add="SOLVED")
            self.solved = True

    def save(self, add=""):
        # Figure out the root of the resulting file names
        if add != "":
            name = "agent_" + add + "_"
        else:
            name = "agent_"

        name = name + "train_time_" + \
            seconds_to_time_str(
                self.data["train_time"]).replace(" ", "_")

        save_path = os.path.join(self.data["folder"], name + ".pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(self.data, f)


def find_state_mag(env, max_steps=1000, n_episodes=1000):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    states = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        state = env_info.vector_observations[0]
        for t in range(max_steps):
            states.append(state)
            actions = np.random.randn(num_agents, action_size)
            actions = np.clip(actions, -1, 1)
            env_info = env.step(actions)[brain_name]
            state = env_info.vector_observations[0]
            done = env_info.local_done[0]
            if done:
                break
    states = np.array(states)
    states = np.abs(states)
    return np.mean(states, axis=0), np.std(states, axis=0)


def seconds_to_time_str(t):
    if t < 0:
        raise Exception("Negative time?")
    if t < 60:
        return "{:02d} seconds".format(int(t))
    elif t >= 60 and t < 3600:
        return "{:04.1f} minutes".format(t/60)
    elif t >= 3600:
        return "{:04.1f} hours".format(t/3600)


def load_agent(folder, add="", train_time="last"):
    if add != "":
        name = "agent_" + add + "_"
    else:
        name = "agent_"

    if train_time != "last":
        name = name + "train_time_" + train_time.replace(" ", "_") + ".pkl"
    else:
        files = glob.glob(os.path.join(folder, "*.pkl"))
        files.sort(key=os.path.getmtime)
        files = files[-1]
        name = os.path.split(files)[1]

    path = os.path.join(folder, name)
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def show_plots(mean_scores, scores, labels=None, max_episodes=None, only_mean=False, legend_outside=False):
    if max_episodes == None:
        # Find max number of episodes
        max_episodes = 0
        for i in range(len(mean_scores)):
            if len(mean_scores[i]) > max_episodes:
                max_episodes = len(mean_scores[i])

    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap("jet", max([len(mean_scores), 2]))
    for i in range(len(mean_scores)):
        if labels is not None:
            label = labels[i]
        else:
            label = None
        mean_score = mean_scores[i]
        score = scores[i]
        if len(mean_score) < max_episodes:
            mean_score = np.concatenate(
                (mean_score, np.nan * np.ones(max_episodes-len(mean_score))))
            score = np.concatenate(
                (score, np.nan * np.ones(max_episodes-len(score))))
        if not only_mean:
            ax.plot(np.arange(1, max_episodes+1),
                    score, alpha=0.3, color=cmap(i))
        ax.plot(np.arange(1, max_episodes+1), mean_score,
                label=label, color=cmap(i), linewidth=2)
    if labels is not None:
        if legend_outside:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend()
    ax.set_xlabel("# episodes")
    ax.grid()
