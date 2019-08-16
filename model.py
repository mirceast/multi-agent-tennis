import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, params):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            params["fc1"] (int): Number of nodes in first hidden layer
            params["fc2"] (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(params["seed"])
        self.params = params

        self.fc1 = nn.Linear(state_size, params["fc1"])
        self.fc2 = nn.Linear(params["fc1"], params["fc2"])
        self.fc3 = nn.Linear(params["fc2"], action_size)

        if params["batchnorm"]:
            self.bn0 = nn.BatchNorm1d(state_size)
            self.bn1 = nn.BatchNorm1d(params["fc1"])
            self.bn2 = nn.BatchNorm1d(params["fc2"])

        self.reset_parameters()

    def reset_parameters(self):
        if self.params["weight_init"].lower() == "uniform":
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        elif self.params["weight_init"].lower() == "kaiming":
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        else:
            raise Exception(
                "Only 'uniform' and 'kaiming' weight inits are supported.")
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if self.params["batchnorm"]:
            x = self.bn0(state)
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = state
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, params):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            params["fc2"] (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(params["seed"])
        self.params = params

        self.fc1 = nn.Linear(state_size, params["fc1"])
        self.fc2 = nn.Linear(params["fc1"] + action_size, params["fc2"])
        self.fc3 = nn.Linear(params["fc2"], 1)

        if params["batchnorm"]:
            self.bn0 = nn.BatchNorm1d(state_size)
            self.bn1 = nn.BatchNorm1d(params["fc1"])

        self.reset_parameters()

    def reset_parameters(self):
        if self.params["weight_init"].lower() == "uniform":
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        elif self.params["weight_init"].lower() == "kaiming":
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        else:
            raise Exception(
                "Only 'uniform' and 'kaiming' weight inits are supported.")
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if self.params["batchnorm"]:
            xs = self.bn0(state)
            xs = F.relu(self.bn1(self.fc1(xs)))
        else:
            xs = state
            xs = F.relu(self.fc1(xs))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
