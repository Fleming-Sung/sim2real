import numpy as np
import torch
from networks import Q_network, Actor, Critic, SAC_Actor, Predictor
from torch.nn import functional as F
import copy
#from torch.utils.tensorboard import SummaryWriter

class Predict_Model(object):

    def __init__(self, state_dim, max_pos):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.state_dim = state_dim
        self.max_pos = max_pos
        self.predictor = Predictor(self.state_dim, max_pos)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=0.01)
        #self.logger = SummaryWriter('./exp_log_tfb')
        self.iter = 0

    def train(self, batch_size, replay_buffer):
        self.iter += 1
        state, next_state = replay_buffer.get(batch_size)
        loss = F.mse_loss(self.predictor.forward(state), next_state)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #self.logger.add_scalar('train_loss', loss.mean().item(), self.iter)



    def save(self, filename):
        torch.save(self.predictor.state_dict(), filename + "_pre")
        torch.save(self.optimizer.state_dict(), filename + "_pre_optimizer")

    def load(self, filename):
        self.predictor.load_state_dict(torch.load(filename + "_pre"))
        self.optimizer.load_state_dict(torch.load(filename + "_pre_optimizer"))


class DQN(object):
    def __init__(self, state_dim, action_num):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        self.state_dim = state_dim
        self.action_num = action_num
        self.Q = Q_network(self.state_dim, self.action_num).to(self.device)
        self.Q_target = Q_network(self.state_dim, self.action_num).to(self.device)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.01)
        self.discount = 0.98
        self.idx = 0
        self.update_freq = 10
        self.tau = 0.01

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        q_t = self.Q(state)
        action_index = torch.argmax(q_t).detach().cpu().numpy().flatten()
        return action_index[0]

    def train(self, replay_buffer, batch_size):
        self.idx += 1
        state, action, next_state, reward, done = replay_buffer.get(batch_size)
        with torch.no_grad():
            next_q, _ = torch.max(self.Q_target(next_state), dim=1)
            next_q = next_q.reshape(-1, 1)
            target_q = reward + self.discount * next_q
        current_q = self.Q(state).gather(dim=1, index=action.long())

        critic_loss = F.mse_loss(current_q, target_q)
        self.Q_optimizer.zero_grad()
        critic_loss.backward()
        self.Q_optimizer.step()

        if self.idx % self.update_freq == 0:
            for param, param_target in zip(self.Q.parameters(),self.Q_target.parameters()):
                param_target.data.copy_(self.tau * param.data + (1 - self.tau) * param_target.data)

    def save(self, filename):
        torch.save(self.Q.state_dict(), filename + "_q")
        torch.save(self.Q_optimizer.state_dict(), filename + "_q_optimizer")

    def load(self, filename):
        self.Q.load_state_dict(torch.load(filename + "_q"))
        self.Q_optimizer.load_state_dict(torch.load(filename + "_q_optimizer"))


class DPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.Actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.Actor_target = copy.deepcopy(self.Actor)
        self.Critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.Critic_target = copy.deepcopy(self.Critic)
        self.actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=1e-4)
        self.discount = 0.98
        self.idx = 0
        self.update_freq = 2
        self.tau = 0.05

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.Actor.forward(state).detach().cpu().numpy().flatten()

    def train(self, replay_buffer, batch_size):
        self.idx += 1
        state, action, next_state, reward, not_done = replay_buffer.get(batch_size)
        with torch.no_grad():
            next_action = self.Actor(next_state)
            next_q = self.Critic(next_state, next_action)
            target_q = reward + self.discount * next_q
        current_q = self.Critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.Critic(state, self.Actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        """
            for param, param_target in zip(self.Critic.parameters(),self.Critic_target.parameters()):
                param_target.data.copy_(self.tau * param.data + (1 - self.tau) * param_target.data)

            for param, param_target in zip(self.Actor.parameters(),self.Actor_target.parameters()):
                param_target.data.copy_(self.tau * param.data + (1 - self.tau) * param_target.data)
         """

    def Q(self, state, action):
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        return self.Critic(state, action).detach().cpu().numpy().flatten()

    def save(self, filename):
        torch.save(self.Critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.Actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.Critic.load_state_dict(torch.load(filename + "_critic",map_location=torch.device('cpu')))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer",map_location=torch.device('cpu')))

        self.Actor.load_state_dict(torch.load(filename + "_actor",map_location=torch.device('cpu')))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer",map_location=torch.device('cpu')))


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.Actor = SAC_Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.Critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=1e-4)
        self.discount = 0.98
        self.alpha = 0.2

    def select_action(self, state, explore=True, get_log_pi=False):
        state = torch.FloatTensor(state).to(self.device)
        action, _ = self.Actor.forward(state, explore, get_log_pi)
        action = action.detach().cpu().numpy().flatten()
        return action

    def train(self, replay_buffer, batch_size):
        state, action, next_state, reward, not_done = replay_buffer.get(batch_size)
        with torch.no_grad():
            next_action, log_pi_next = self.Actor(next_state)
            next_q = self.Critic(next_state, next_action)
            target_q = reward + self.discount * (next_q - self.alpha * log_pi_next)
        current_q = self.Critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        act, log_pi = self.Actor(state)
        actor_loss = (self.alpha * log_pi - self.Critic(state, act)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def Q(self, state, action):
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        return self.Critic(state, action).detach().cpu().numpy().flatten()

    def save(self, filename):
        torch.save(self.Critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.Actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.Critic.load_state_dict(torch.load(filename + "_critic",map_location=torch.device('cpu')))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer",map_location=torch.device('cpu')))

        self.Actor.load_state_dict(torch.load(filename + "_actor",map_location=torch.device('cpu')))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer",map_location=torch.device('cpu')))
