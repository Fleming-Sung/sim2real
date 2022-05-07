import torch
import numpy as np


def softmax(x):
    row_max = np.max(x)
    x -= row_max
    x_exp = np.exp(x)
    s = x_exp / np.sum(x_exp, keepdims=True)
    return s


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))
        self.num = 0
        self.pt = 0
        self.max_size = max_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

    def add(self, state, action, next_state, reward, done):
        self.state[self.pt] = state
        self.action[self.pt] = action
        self.next_state[self.pt] = next_state
        self.reward[self.pt] = reward
        self.done[self.pt] = done
        self.pt = (self.pt + 1) % self.max_size
        self.num = self.num + 1

    def priority_get(self, batch_size):
        priority = softmax(self.reward.reshape(1, -1)[0][0: min(self.num, self.max_size)])
        seq = np.arange(0, min(self.num, self.max_size), 1)
        idx = np.random.choice(seq, size=batch_size, p=priority)
        return torch.FloatTensor(self.state[idx]).to(self.device), \
               torch.FloatTensor(self.action[idx]).to(self.device), \
               torch.FloatTensor(self.next_state[idx]).to(self.device), \
               torch.FloatTensor(self.reward[idx]).to(self.device), \
               torch.FloatTensor(self.done[idx]).to(self.device)
    def get(self, batch_size):
        # centroid = np.mean(self.state, axis=0)
        # priority = softmax(np.sum((self.state - centroid)**2, axis=1))
        # idx = np.argsort(-priority)[:batch_size]
        idx = np.random.randint(0, min(self.num, self.max_size), batch_size)
        return torch.FloatTensor(self.state[idx]).to(self.device), \
               torch.FloatTensor(self.action[idx]).to(self.device), \
               torch.FloatTensor(self.next_state[idx]).to(self.device),\
               torch.FloatTensor(self.reward[idx]).to(self.device), \
               torch.FloatTensor(self.done[idx]).to(self.device)

if __name__ == "__main__":
    A = np.array([[1,2],[2,3],[3,4],[4,5]])
    coord = A[:, 0]
    centroid = np.mean(coord)
    print(centroid)
    priority = softmax((coord - centroid) ** 2)
    print(priority)
    seq = np.arange(0, 4, 1)
    idx = np.random.choice(seq, size=2, p=priority)
    print(idx)