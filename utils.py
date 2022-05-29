import torch
import numpy as np
from tool_box import dist_Gaussian

def softmax(x):
    row_max = np.max(x)
    x -= row_max
    x_exp = np.exp(x)
    s = x_exp / np.sum(x_exp, keepdims=True)
    return s

def evaluation(env, policy, ep_num = 10):
    eva_reward = 0
    for j in range(ep_num):
        observation, done = env.reset(), False
        goal = 0
        state = get_state(observation, goal)
        print(observation["vector"][5 + goal][0:2])
        while not (done or goal > 4):
            action = policy.select_action(state)
            mx, my, dtheta = action[0], action[1], action[2]
            action_take = [mx,my, dtheta, 0]

            next_observation, reward, done, info = env.step(action_take)

            if next_observation["vector"][5 + goal][2]:
                goal = goal + 1
                print(next_observation["vector"][5 + goal][0:2])
            next_state = get_state(next_observation, goal)
            observation = next_observation
            state = next_state
            eva_reward += reward
    return eva_reward/ep_num

def get_theta(o, g):
    '''
    计算 小车朝向 与 目标 之间的夹角
    '''
    vector = o["vector"]
    self_pose = vector[0]  # [x, y, theta(rad)]
    goal_pose = vector[5 + g]  # [x, y, is_activated?]

    tanTheta = (goal_pose[1] - self_pose[1]) / (goal_pose[0] - self_pose[0])

    # case 1
    if ((goal_pose[1] - self_pose[1] >= 0) and (goal_pose[0] - self_pose[0] >= 0)):
        if ((self_pose[2] <= 0) or ((self_pose[2] >= 0) and (self_pose[2] <= np.pi + np.arctan(tanTheta)))):
            theta = np.arctan(tanTheta) - self_pose[2]
        else:
            theta = np.arctan(tanTheta) - self_pose[2] + 2 * np.pi
    # case 2
    elif ((goal_pose[1] - self_pose[1] >= 0) and (goal_pose[0] - self_pose[0] < 0)):
        if (((self_pose[2] <= 0) and (self_pose[2] >= np.arctan(tanTheta))) or (self_pose[2] >= 0)):
            theta = np.pi + np.arctan(tanTheta) - self_pose[2]
        else:
            theta = np.arctan(tanTheta) - self_pose[2] - np.pi
    # case 3
    elif ((goal_pose[1] - self_pose[1] < 0) and (goal_pose[0] - self_pose[0] < 0)):
        if ((self_pose[2] <= 0) or ((self_pose[2] >= 0) and (self_pose[2] <= np.arctan(tanTheta)))):
            theta = np.arctan(tanTheta) - self_pose[2] - np.pi
        else:
            theta = np.arctan(tanTheta) - self_pose[2] + np.pi
    # case 4
    else:
        if (((self_pose[2] <= 0)) or ((self_pose[2] >= 0) and (self_pose[2] <= np.pi + np.arctan(tanTheta)))):
            theta = np.arctan(tanTheta) - self_pose[2]
        else:
            theta = np.arctan(tanTheta) - self_pose[2] + 2 * np.pi

    return theta


def get_state(o, g):
    '''
    从 obs 中计算 状态 [self_x,self_y,goal_x,goal_y,theta,goal_index]
    '''
    vector = o["vector"]
    laser = o["laser"]

    left_laser = laser[10]
    mid_laser = laser[30]
    right_laser = laser[50]

    # (-135, 135), 61 points, resolution 270/(61 -1), 90/resolution = 20.
    self_pose = vector[0]  # [x, y, theta(rad)]
    goal_pose = vector[5 + g]  # [x, y, is_activated?]

    dis = np.sqrt((self_pose[0] - goal_pose[0]) ** 2 + (self_pose[1] - goal_pose[1]) ** 2)  # 距离
    # dis = abs(self_pose[0] - goal_pose[0]) + abs(self_pose[1] - goal_pose[1])  # 距离
    theta = get_theta(o, g)  # 夹角角度

    # theta = (o[4 + g][0] - o[1][0])/dis - np.cos(np.pi + o[1][2]) + (o[4 + g][1] - o[1][1])/dis - np.sin(np.pi + o[1][2])
    # return [self_pose[0], self_pose[1], self_pose[2],  get_dist(o, g), get_theta(o, g),
    #        vector[5][0], vector[5][1], vector[6][0],  vector[6][1],
    #        vector[7][0], vector[7][1], vector[8][0],  vector[8][1],
    #        vector[9][0], vector[9][1]]
    # return [o[1][0], o[1][1], o[4+g][0], o[4+g][1], o[1][2], g]
    return [laser[0], laser[10], laser[20], laser[30], laser[40], laser[50], laser[60], dis, get_theta(o, g)]

def get_dist(o, g):
    '''
    小车与目标的距离 + 剩余所有目标之间的的顺次距离之和
    '''
    vector = o["vector"]
    self_pose = vector[0]  # [x, y, theta(rad)]
    goal_pose = vector[5 + g]  # [x, y, is_activated?] 当前目标
    dis = np.sqrt((self_pose[0] - goal_pose[0])**2 + (self_pose[1] - goal_pose[1])**2) # 距离
    # dis = abs(self_pose[0] - goal_pose[0]) + abs(self_pose[1] - goal_pose[1])  # 距离

    # dis = dist_Gaussian(np.array(self_pose[0:2]), np.array(goal_pose[0:2]), mean, covar, amp)
    # for j in (g, 4):
    #     dis += np.sqrt( (vector[5+g+1][0] - vector[5 + g][0])**2 + (vector[5+g+1][1] - vector[5 + g][1])**2 )
    return dis

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