# sim2real
UCAS course[Reinforcement Learning] group assignment, as well as  the materious for COG competition 2022.

## 2022.05.08
离散的DQN

CogEnvDecoder == 0.1.29

discrete_v2.py 中 修改了小车朝向与目标之间的夹角；奖励函数是 欧氏距离+碰撞次数+碰撞时间；训练过程中每找到一个目标就reset。

test_env.py 是直接evaluate上面discrete_v2.py中的最好的模型

## 2022.05.14

continue.py 三个动作 dx  dy  dtheta; 奖励函数：goal_reward/ dis_reward/ collision_reward； 训练方式：遇到障碍物就reset； 每局训练

存在问题：有时会卡在某一个地方。

## 2022.05.18

continue_v3.py 设置卡顿标志，加入临时goal：卡顿后，有30步前往临时目标； 每步训练

test_env_continue.py 环境测试 v2版本，要提前在官网的 tag 里下载好

## 2022.05.19

continue_v4.py: update the flag of stuck. 


## 2022.05.22
Toolbox.py 中提供了更新的计算距离的方式：高斯混合模型模拟障碍物 曲线积分计算实际距离
