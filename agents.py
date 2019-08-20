from OUNoise import OUNoise
from torch.optim import Adam
from utilities import hard_update, gumbel_softmax, onehot_from_logits
import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from utilities import get_env_info, transpose_to_tensor, transpose_list, soft_update

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgentDDPG():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, params, n_agents=2):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            params (dict): all parameters
        """
        self.params = params
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(params["seed"])

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(
            state_size, action_size, params).to(device)
        self.actor_target = Actor(
            state_size, action_size, params).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=params["lr_actor"])

        # Critic Network (w/ Target Network)
        if params["type"] == "MADDPG":
            self.critic_local = Critic(
                n_agents * state_size, n_agents * action_size, params).to(device)
            self.critic_target = Critic(
                n_agents * state_size, n_agents * action_size, params).to(device)
            self.critic_optimizer = optim.Adam(
                self.critic_local.parameters(), lr=params["lr_critic"], weight_decay=params["weight_decay"])
        else:
            self.critic_local = Critic(
                state_size, action_size, params).to(device)
            self.critic_target = Critic(
                state_size, action_size, params).to(device)
            self.critic_optimizer = optim.Adam(
                self.critic_local.parameters(), lr=params["lr_critic"], weight_decay=params["weight_decay"])

        # initialize targets same as original networks
        hard_update(self.actor_target, self.actor_local)
        hard_update(self.critic_target, self.critic_local)

        # Noise process
        self.noise = OUNoise(action_size, params["seed"])

        # Replay memory
        self.memory = ReplayBuffer(
            action_size, self.params["buffer_size"], self.params["batch_size"], params["seed"])

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        try:
            n_obs = len(reward)
        except TypeError:
            n_obs = 1

        if n_obs == 1:
            self.memory.add(state, action, reward, next_state, done)
        else:
            for i in range(n_obs):
                self.memory.add(state[i], action[i],
                                reward[i], next_state[i], done[i])

        # Learn, if enough samples are available in memory
        self.t_step = (self.t_step + 1) % self.params["update_every"]
        if self.t_step == 0:
            if len(self.memory) > self.params["batch_size"]:
                for i in range(self.params["update_steps"]):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.params["gamma"])

    def act(self, state, add_noise=True, no_grad=True):
        """Returns actions for given state as per current policy."""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if no_grad:
            self.actor_local.eval()
            with torch.no_grad():
                action = self.actor_local(state)  # .cpu().data.numpy()
            self.actor_local.train()
        else:
            action = self.actor_local(state)  # .cpu().data.numpy()
        if add_noise:
            action += torch.from_numpy(self.noise.sample()).float().to(device)
        return torch.clamp(action, -1, 1)

    def target_act(self, state, add_noise=True, no_grad=True):
        """Returns actions for given state as per current policy."""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if no_grad:
            self.actor_target.eval()
            with torch.no_grad():
                action = self.actor_target(state)  # .cpu().data.numpy()
            self.actor_target.train()
        else:
            action = self.actor_target(state)  # .cpu().data.numpy()
        if add_noise:
            action += torch.from_numpy(self.noise.sample()).float().to(device)
        return torch.clamp(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        if not self.params["double"]:
            actions_next = self.actor_target(next_states)
        else:
            actions_next = self.actor_local(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local,
                         self.critic_target, self.params["tau"])
        self.soft_update(self.actor_local, self.actor_target,
                         self.params["tau"])

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)


class AgentMADDPG:
    def __init__(self, env, params):
        super(AgentMADDPG, self).__init__()
        self.params = params

        brain_name, n_agents, state_size, action_size = get_env_info(env)
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.maddpg_agent = [AgentDDPG(
            state_size=state_size, action_size=action_size, params=params, n_agents=self.n_agents) for _ in range(self.n_agents)]

        # Noise process
        self.noise = OUNoise(action_size, params["seed"])

        # Replay memory
        self.memory = ReplayBuffer(
            action_size, self.params["buffer_size"], self.params["batch_size"], params["seed"])

        self.t_step = 0

        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor_local for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [
            ddpg_agent.actor_target for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, no_grad=True):
        """get actions from all agents in the MADDPG object"""
        # Reshape to be consistent with target_act
        if obs_all_agents.shape[1] != self.state_size * self.n_agents:
            obs_all_agents = obs_all_agents.reshape(
                -1, self.state_size * self.n_agents)
        actions = []
        for i, agent in enumerate(self.maddpg_agent):
            obs = self.select_agent_state(obs_all_agents, i)
            actions.append(agent.act(obs, no_grad=no_grad))
        actions = torch.cat(actions, dim=1)
        return actions

    def target_act(self, obs_all_agents, no_grad=True):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = []
        for i, agent in enumerate(self.maddpg_agent):
            obs = self.select_agent_state(obs_all_agents, i)
            target_actions.append(agent.target_act(obs, no_grad=no_grad))
        target_actions = torch.cat(target_actions, dim=1)
        return target_actions

    def get_agent_state_idx(self, agent_number):
        return int(agent_number * self.state_size), int((agent_number + 1) * self.state_size)

    def get_agent_action_idx(self, agent_number):
        return int(agent_number * self.action_size), int((agent_number + 1) * self.action_size)

    def select_agent_state(self, concatenated_states, agent_number):
        idx1, idx2 = self.get_agent_state_idx(agent_number)
        return concatenated_states[:, idx1:idx2]

    def select_agent_action(self, concatenated_actions, agent_number):
        idx1, idx2 = self.get_agent_action_idx(agent_number)
        return concatenated_actions[:, idx1:idx2]

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Let's see if flattening everything helps
        state = state.reshape(-1, self.n_agents * self.state_size)
        action = action.reshape(-1, self.n_agents * self.action_size)
        next_state = next_state.reshape(-1, self.n_agents * self.state_size)
        # And just to be safe, make sure that everything has the same shape
        reward = np.expand_dims(np.array(reward), axis=1).T
        done = np.expand_dims(np.array(done), axis=1).T

        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        self.t_step = (self.t_step + 1) % self.params["update_every"]
        if self.t_step == 0:
            if len(self.memory) > self.params["batch_size"]:
                for i in range(self.params["update_steps"]):
                    self.learn()

    def reset(self):
        for agent in self.maddpg_agent:
            agent.reset()

    def learn(self):
        for a_i in range(self.n_agents):
            experiences = self.memory.sample()
            self.update(experiences, a_i)
        self.update_targets()  # soft update the target network towards the actual networks

    def update(self, experiences, agent_number):
        """update the critics and actors of all the agents """

        states, actions, rewards, next_states, dones = experiences

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # I don't know why, but it wasn't working with calling self.target_act() or .act()
        obs_all = [states[:, 0:24], states[:, 24:]]
        next_obs_all = [next_states[:, 0:24], next_states[:, 24:]]
        if not self.params["double"]:
            next_actions = [self.maddpg_agent[i].actor_target(
                obs) for i, obs in enumerate(next_obs_all)]
            # next_actions = self.target_act(next_states, no_grad=False)
        else:
            next_actions = [self.maddpg_agent[i].actor_local(
                obs) for i, obs in enumerate(next_obs_all)]

        with torch.no_grad():
            q_next = agent.critic_target(torch.cat(next_obs_all, dim=1),
                                         torch.cat(next_actions, dim=1)).squeeze()

        y = rewards[:, agent_number] + self.params["gamma"] * \
            q_next * (1 - dones[:, agent_number])
        q = agent.critic_local(torch.cat(obs_all, dim=1), actions).squeeze()

        # huber_loss = torch.nn.SmoothL1Loss()
        # critic_loss = huber_loss(q, y.detach())
        critic_loss = F.mse_loss(q, y)
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()

        # get the policy gradient
        # again, for some reason, self.act() wasn't working - gave a backward error
        # local_actions = self.act(states, no_grad=False)

        local_actions = [self.maddpg_agent[i].actor_local(
            obs) for i, obs in enumerate(obs_all)]

        actor_loss = - \
            agent.critic_local(torch.cat(obs_all, dim=1),
                               torch.cat(local_actions, dim=1)).mean()
        # actor_loss = -agent.critic_local(states, local_actions).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.actor_target,
                        ddpg_agent.actor_local, self.params["tau"])
            soft_update(ddpg_agent.critic_target,
                        ddpg_agent.critic_local, self.params["tau"])


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
