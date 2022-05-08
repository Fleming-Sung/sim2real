from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import numpy as np
import algorithm
import utils
import time

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

    dis = np.sqrt((self_pose[0] - goal_pose[0]) ** 2 + (self_pose[1] - goal_pose[1]) ** 2)  # 距离

    theta = get_theta(o, g)  # 夹角角度

    # theta = (o[4 + g][0] - o[1][0])/dis - np.cos(np.pi + o[1][2]) + (o[4 + g][1] - o[1][1])/dis - np.sin(np.pi + o[1][2])
    return [self_pose[0], self_pose[1], goal_pose[0], goal_pose[1], theta, g]
    # return [o[1][0], o[1][1], o[4+g][0], o[4+g][1], o[1][2], g]


def evaluation(env, policy):
    observation, done = env.reset(), False
    eva_reward = 0
    goal = 0
    state = get_state(observation, goal)
    for j in range(1000):
        print(state[2],state[3])
        action_index = policy.select_action(state)
        mx ,my = action_space[action_index]
        action_take = [mx,my,  state[4]/np.pi, 0]

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



env = CogEnvDecoder(env_name="win_V1/RealGame.exe", no_graphics=False, time_scale=1, worker_id=1) # windows os
num_episodes = 10
num_steps_per_episode = 100 # max: 1500

action_dim = 1
action_space = [(0, 0.5), (0, -0.5), (0.5, 0), (-0.5, 0)] #
state_dim = 6  #sx, sy, goalx, goaly, theta, goal
policy = algorithm.DQN(state_dim, len(action_space))
policy.load("model")



for i in range(num_episodes):
    #every time call the env.reset() will reset the envinronment
    observation = env.reset()
    eva_reward = evaluation(env, policy)
    print(f"Evaluation:{eva_reward}-------------------------------------------")