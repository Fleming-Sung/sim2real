import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
import numpy as np


class Q_network(nn.Module):
    def __init__(self, state_dim, action_num):
        super(Q_network, self).__init__()
        self.l1 = nn.Linear(state_dim, 16)
        self.l2 = nn.Linear(16, 32)
        self.l3 = nn.Linear(32, action_num)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 16)
        self.l2 = nn.Linear(16, 32)
        self.l3 = nn.Linear(32, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return self.max_action * torch.tanh(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 16)
        self.l2 = nn.Linear(16, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class SAC_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(SAC_Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 16)
        self.l2 = nn.Linear(16, 32)
        self.l3 = nn.Linear(32, 16)
        self.mu_layer = nn.Linear(16, action_dim)
        self.log_std_layer = nn.Linear(16, action_dim)
        self.max_action = max_action

    def forward(self, state, explore=True, get_log_pi=True):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std)
        distribution = Normal(mu, std)
        if explore:
            act = distribution.rsample()
        else:
            act = mu
        # Quote from DRL_OpenAI by OpenAi spinning-up
        if get_log_pi:
            log_pi = distribution.log_prob(act).sum(axis=-1) - (2 * (np.log(2) - act - F.softplus(-2 * act))).sum(axis=1)
            log_pi = log_pi.reshape(-1, 1)
        else:
            log_pi = None

        return self.max_action * torch.tanh(act), log_pi