#pip install CogEnvDecoder==0.1.0

from Cogenvdecoder import CogEnvDecoder
import numpy as np
import algorithm
import utils

def evaluation(env, policy):
    observation, done = env.reset(), False
    eva_reward = 0
    goal = 0
    state = get_state(observation, goal)
    for i in range(1000):
        theta = np.arctan((state[3] - state[1])/(state[2] - state[0]))
        action_index = policy.select_action(state)
        mx ,my = action_space[action_index]
        action = [mx,my, (theta - observation[1][2]) / (0.5 * np.pi),0]

        next_observation, reward, done, info = env.step(action)
        if observation[initial_goal + goal][2]:
            goal = goal + 1
        next_state = get_state(next_observation, goal)
        observation = next_observation
        state = next_state
        eva_reward += reward
        if goal > 4:
            break
    return eva_reward
def get_state(o, g):

    
    dis = np.sqrt((o[4 + g][0]-o[1][0])**2 + (o[4 + g][1]-o[1][1])**2)
    tanTheta = (o[1][1] - o[4 + g][1]) / (o[1][0] - o[4 + g][0])
    theta = np.arctan(tanTheta) - o[1][2]
    # theta = (o[4 + g][0] - o[1][0])/dis - np.cos(np.pi + o[1][2]) + (o[4 + g][1] - o[1][1])/dis - np.sin(np.pi + o[1][2])
    return [o[1][0], o[1][1], o[4+g][0], o[4+g][1], o[1][2], g]

def get_dist(o, g):
    dis = np.sqrt((o[4 + g][0]-o[1][0])**2 + (o[4 + g][1]-o[1][1])**2)
    for j in (g, 4):
        dis += np.sqrt((o[5 + j][0]-o[4 + j][0])**2 + (o[5 + j][1]-o[4 + j][1])**2)
    return dis

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
if __name__ == "__main__":
    env = CogEnvDecoder.CogEnvDecoder(env_name='1.x86_64')
    obs = env.reset()
    
    action_dim = 1
    action_space = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    state_dim = 6  #sx, sy, goalx, goaly, theta, goal
    policy = algorithm.DQN(state_dim, len(action_space))
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
        idx += 1
        if i < max_explore:
            action_index = np.random.choice(range(len(action_space)))
        else:
            if np.random.uniform()>eipsilon:
                action_index = policy.select_action(state)
            else:
                action_index = np.random.choice(range(len(action_space)))


        dx, dy =  action_space[action_index]
        bool_done = False
        dtheta = (obs[1][2] - np.arctan((state[3] - state[1])/(state[2] - state[0])))/(0.5 * np.pi)
        dbullet = 0
        action = [dx, dy, dtheta, dbullet]
        next_obs, reward, done, info = env.step(action)  #算法输出动作与环境交互，得到新的观测量



        next_state = get_state(next_obs, goal)
        reward = reward - get_dist(next_obs, goal) - 10 * (next_obs[9][1] - obs[9][1])

        if next_obs[4 + goal][2]:
            goal = goal + 1
            if goal > 4:
                bool_done = True
        next_state = get_state(next_obs, goal)
        replay_buffer.add(state, action_index, next_state, reward, done)
        state = next_state
        obs = next_obs
        episode_reward += reward
        
        if idx > 999 or bool_done:

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

        if i> max_explore:
            policy.train(replay_buffer, batch_size)
        """
        img = observation[0]                           #摄像头所获取到的图像信息
        
        sx = observation[1][0]                         #主车x方向位置、单位：m
        sy = observation[1][1]                         #主车y方向位置、单位：m
        syaw = observation[1][2]                       #主车在世界坐标系下的角度、单位：弧度
       

        esx = observation[2][0]                        #敌车x方向位置、单位：m
        esy = observation[2][1]                        #敌车y方向位置、单位：m
        eyaw = observation[2][2]                       #敌车在世界坐标系下的角度、单位：弧度

        HP = observation[3][0]                         #主车的血量
        Bullet = observation[3][1]                     #主车的弹药

        #五个目标点的位置随机给出
        goal_pos1 = observation[4]                     #x:[4][0] y:[4][1] 是否被激活：[4][2]
        goal_pos2 = observation[5]                     #x:[5][0] y:[5][1] 是否被激活：[5][2]
        goal_pos3 = observation[6]                     #x:[6][0] y:[6][1] 是否被激活：[6][2]
        goal_pos4 = observation[7]                     #x:[7][0] y:[7][1] 是否被激活：[7][2]
        goal_pos5 = observation[8]                     #x:[8][0] y:[8][1] 是否被激活：[8][2]

        collision_times = observation[9][0]            #碰撞次数
        collision_time = observation[9][1]             #碰撞时间
        """
        
