#pip install CogEnvDecoder==0.1.0

from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import numpy as np
import algorithm
import utils
import time

def evaluation(env, policy):
    observation, done = env.reset(), False
    eva_reward = 0
    goal = 0
    state = get_state(observation, goal)
    for j in range(1000):

        theta = np.arctan((state[3] - state[1])/(state[2] - state[0]))
        action = policy.select_action(state)
        mx ,my = action[0], action[1]
        action_take = [mx,my, state[4]/np.pi, 0]

        next_observation, reward, done, info = env.step(action_take)

        if next_observation["vector"][5 + goal][2]:
            goal = goal + 1
        next_state = get_state(next_observation, goal)
        observation = next_observation
        state = next_state
        eva_reward += reward
        if goal > 4:
            break
    return eva_reward


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
    self_pose = vector[0]  # [x, y, theta(rad)]
    goal_pose = vector[5 + g]  # [x, y, is_activated?]
    
    dis = np.sqrt((self_pose[0] - goal_pose[0])**2 + (self_pose[1] - goal_pose[1])**2) # 距离

    theta = get_theta(o, g) # 夹角角度

    # theta = (o[4 + g][0] - o[1][0])/dis - np.cos(np.pi + o[1][2]) + (o[4 + g][1] - o[1][1])/dis - np.sin(np.pi + o[1][2])
    return [self_pose[0], self_pose[1], goal_pose[0], goal_pose[1], theta, g]
    # return [o[1][0], o[1][1], o[4+g][0], o[4+g][1], o[1][2], g]

def get_dist(o, g):
    '''
    小车与目标的距离 + 剩余所有目标之间的的顺次距离之和
    '''
    vector = o["vector"]
    self_pose = vector[0]  # [x, y, theta(rad)]
    goal_pose = vector[5 + g]  # [x, y, is_activated?] 当前目标
    dis = np.sqrt((self_pose[0] - goal_pose[0])**2 + (self_pose[1] - goal_pose[1])**2) # 距离
    for j in (g, 4):
        dis += np.sqrt( (vector[5+g+1][0] - vector[5 + g][0])**2 + (vector[5+g+1][1] - vector[5 + g][1])**2 )
    return dis

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


if __name__ == "__main__":

    env = CogEnvDecoder(env_name="win_V1/RealGame.exe", no_graphics=False, time_scale=1, worker_id=1) # windows os

    obs = env.reset()

    # 等待 环境加载完成
    while(len(obs["color_image"].shape) == 2):
        obs = env.reset()

    print('环境加载完成')
    action_dim = 2
    max_action = 1
    # action_space = [(0, 1), (0, -1), (1, 0), (-1, 0)] #
    state_dim = 6  #sx, sy, goalx, goaly, theta, goal
    policy = algorithm.DPG(state_dim, action_dim, max_action)
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    batch_size = 256
    episode_reward = 0
    max_explore = 100
    idx = 0
    episode = 0
    initial_goal = 4
    goal = 0
    eipsilon = 0.05
    state = get_state(obs, goal)

    for i in range(1000000):
        print(state)

        if i < max_explore:
            action = np.random.uniform(-max_action, max_action, action_dim)
        else:
            if np.random.uniform() > eipsilon:
                action = policy.select_action(state)
            else:
                action = np.random.uniform(-max_action, max_action, action_dim)



        dx,dy = action[0],action[1]
        bool_done = False # 是否完全找到五个目标
        dtheta = state[4] / np.pi # state[4] 是夹角， 除以 np.pi 是归一化到 -1 ~ 1
        dbullet = 0 # 是否 射击
        action_take = [dx, dy, dtheta, dbullet] # 动作
        next_obs, reward, done, info = env.step(action_take)  #算法输出动作与环境交互，得到新的观测量

        next_state = get_state(next_obs, goal)  # 根据交互后新的观测量 得到下一个状态

        vector_data = obs["vector"]
        next_vector_data = next_obs["vector"]

        # 碰撞次数 vector_data[10][0] 碰撞时刻 vector_data[10][1]
        reward = reward - get_dist(next_obs, goal) - 50 * (next_vector_data[10][1] - vector_data[10][1]) - 20 * (next_vector_data[10][0] - vector_data[10][0])

        # 如果到达当前 goal 那么就把 goal 定为下一个
        if next_vector_data[5 + goal][2]:
            goal = goal + 1
            if goal > 4:
                bool_done = True

        # 将采样到的数据放入回放池
        replay_buffer.add(state, action, next_state, reward, done)
        # 状态更新
        state = next_state
        obs = next_obs
        episode_reward += reward

        # 当前局结束
        if idx > 999 or bool_done or done:
            episode = episode + 1
            print(f"Episode Num:{episode}, Total:{idx}, Reward:{episode_reward}")
            idx = 0
            episode_reward = 0
            if episode % 5 == 0:
                eva_reward = evaluation(env, policy)
                print(f"Evaluation:{eva_reward}-------------------------------------------")
            obs, done = env.reset(), False
            goal = 0
            state = get_state(obs, goal)

        if i > max_explore:
            policy.train(replay_buffer, batch_size)
        
