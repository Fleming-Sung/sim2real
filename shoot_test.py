from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import numpy as np
import algorithm
from utils import get_state, get_theta, get_dist, evaluation
import utils
import time


env = CogEnvDecoder(env_name="win_confrontation_v2.1/cog_confrontation_env.exe", no_graphics=False, time_scale=1, worker_id=1) # windows os
num_episodes = 10
num_steps_per_episode = 100 # max: 1500

action_dim = 3
#action_space = [(0, 1), (0, -1), (1, 0), (-1, 0)] #
state_dim = 9  #sx, sy, goalx, goaly, theta, goal
policy = algorithm.DPG(state_dim, action_dim , 1)
policy.load("2_1_reward_model/eva_reward_max")
epsilon = 0.95
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def func(x):
    if x > 0:
        return np.pi * np.log2(1 + x)/np.log2(1 + np.pi)
    else:
        return -np.pi * np.log2(1 - x) / np.log2(1 + np.pi)

shoot_policy = algorithm.DQN(3, 2)
replay_buffer = utils.ReplayBuffer(3, 1)
obs, done = env.reset(), False
max_action = 1.0
goal = -2
while (len(obs["color_image"].shape) == 2):
    obs = env.reset()
print('环境加载完成')
for i in range(100000):

    state = get_state(obs, goal)
    #print(state[8])
    #print(func(state[8]))
    #print(func(state[8]) ** (0 + state[8]>0) * (-func(-state[8])) ** (1.0-(state[8]>0)))
    shoot_state = [state[3], state[7], state[8]]
    modified_state = [state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7] * sigmoid(state[7] - 5), func(state[8])]
    if state[7] > 3 or state[8]>0.01:
        action = (policy.select_action(modified_state) + np.random.normal(0, 0.5, action_dim)).clip(-max_action, max_action)
    else:
        theta = state[8] + obs["vector"][0][2] - 0.5 * np.pi
        forward = np.sin(theta)
        turn = np.cos(theta)
        action = [0, np.random.choice([-1, 1]), 0]
    if np.random.uniform()>epsilon:
        shoot = np.random.choice([0,1])
    else:
        shoot = shoot_policy.select_action(shoot_state)
    mx, my, dtheta = action[0], action[1], action[2]
    action_take = [mx, my, dtheta, shoot]
    next_obs, reward, done, info = env.step(action_take)
    next_state = get_state(next_obs, goal)
    next_shoot_state = [next_state[3], next_state[7], next_state[8]]
    vector_data = obs["vector"]
    next_vector_data = next_obs["vector"]
    reward = vector_data[4][0] - next_vector_data[4][0] - 10 * (vector_data[1][1] - next_vector_data[1][1])
    replay_buffer.add(shoot_state, shoot, next_shoot_state, reward, done)
    obs = next_obs
    state = next_state
    if i % 500 == 0 or done:
        print(800.0 - obs["vector"][4][0] + obs["vector"][1][0])
        obs, done = env.reset(), False

        shoot_policy.save("shoot")
    if i > 2000:
        shoot_policy.train(replay_buffer, 256)

# for i in range(num_episodes):
#     #every time call the env.reset() will reset the envinronment
#     observation = env.reset()
#     eva_reward = evaluation(env, policy,1)
#     print(f"Evaluation:{eva_reward}-------------------------------------------")