from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import numpy as np
import algorithm
import utils
import time
from continue_v2 import get_theta, get_dist, get_state
import torch

def evaluation(env, policy):
    observation, done = env.reset(), False
    eva_reward = 0
    goal = 0
    state = get_state(observation, goal)
    print(observation["vector"][5 + goal])
    for j in range(500):
        action = policy.select_action(state)
        mx, my, dtheta = action[0], action[1], action[2]
        action_take = [mx, my, dtheta, 0]
        
        next_observation, reward, done, info = env.step(action_take)
        state_tensor = torch.FloatTensor(state).reshape(1,-1).to(torch.device('cpu'))
        action_tensor = torch.FloatTensor(action_take[0:3]).reshape(1,-1).to(torch.device('cpu'))

        
        # print(get_dist(observation, goal))

        # print(state_tensor.shape, action_tensor.shape)
        # print(next_observation["vector"][5 + goal], action_take, policy.Q(state_tensor, action_tensor))

        if next_observation["vector"][5 + goal][2]:
            goal = goal + 1
            print(next_observation["vector"][5 + goal])
        next_state = get_state(next_observation, goal)
        observation = next_observation
        state = next_state
        eva_reward += reward
        if goal > 4 or done:
            break
    return eva_reward


if __name__ == "__main__":
    env = CogEnvDecoder(env_name="win_v2/cog_sim2real_env.exe", no_graphics=False, time_scale=1, worker_id=1)  # windows os
    num_episodes = 10

    obs = env.reset()

    # 等待 环境加载完成
    while (len(obs["color_image"].shape) == 2):
        obs = env.reset()

    print('环境加载完成')

    action_dim = 3
    max_action = 1.0
    # action_space = [(0, 1), (0, -1), (1, 0), (-1, 0)] #
    state_dim = 9  # ll,ml,rl,dist,theta
    policy = algorithm.DPG(state_dim, action_dim, max_action)
    policy.load("eva_reward_max")

    

    for i in range(num_episodes):
        # every time call the env.reset() will reset the envinronment
        observation = env.reset()
        eva_reward = evaluation(env, policy)
        print(f"Evaluation:{eva_reward}-------------------------------------------")