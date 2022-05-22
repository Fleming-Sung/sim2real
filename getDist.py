import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


"""
1. 给定障碍物 2 个点，确定 mu sigma 幅值
2. 生成GMM
3. 计算距离：根据GMM的表达式 进行曲线积分
    3.1 start：(x1,y1,z1)  goal: (x2,y2,z2) 
    3.2 z = GMM(x,y)  在障碍物附近的值很大; z1 = z2 = 0
    3.2 计算 曲线积分
"""
def get_dist(start,goal, mean_list, cov_list, amp):
    step = 100
    x = np.arange(start[0], goal[0], (abs(start[0]-goal[0]))/step)
    y = np.arange(start[1], goal[1], (abs(start[1]-goal[1]))/step)
    dist = 0
    for i in range(step-1):
        z = GaussMixture(np.array([x[i], y[i]]), mean_list, cov_list, amp)
        z_ = GaussMixture(np.array([x[i+1], y[i+1]]), mean_list, cov_list, amp)
        dz = z_ - z
        temp = np.power(x[i+1] - x[i], 2) + np.power(y[i+1] - y[i], 2) + np.power(dz, 2)
        dist += np.sqrt(temp)

        ax.scatter(x[i],y[i], z)

    return dist


def GaussMixture(x, mean_list, cov_list, amp):

    k = len(mean_list)
    prob = 0
    for i in range(k):
        prob += (1/k) * Gaussian(x, mean_list[i], cov_list[i])
    return prob * amp

def Gaussian(x,mean,cov):

    """
    这是自定义的高斯分布概率密度函数
    :param x: 输入数据
    :param mean: 均值数组
    :param cov: 协方差矩阵
    :return: x的概率
    """

    dim = np.shape(cov)[0]
    # cov的行列式为零时的措施
    covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
    covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
    xdiff = (x - mean).reshape((1,dim))
    # 概率密度
    temp1 = np.power(np.power(2*np.pi,dim)*np.abs(covdet),0.5)
    temp2 = np.exp(-0.5*xdiff.dot(covinv).dot(xdiff.T))

    prob = 1.0/temp1 * temp2[0][0]
    return prob


if __name__ == "__main__":
    X = np.arange(0, 8, 0.1)
    Y = np.arange(0, 4, 0.1)

    x = np.array([1,1])
    mean_list = [np.array([5,2.5]), np.array([1,1])]
    cov_list = [np.array([[0.1,0],[0,0.1]]), np.array([[0.1,0],[0,0.1]])]

    
    X_, Y_ = np.meshgrid(X, Y)
    print(X_.shape)
    print(Y_.shape)

    Z = np.zeros(X_.shape)
    print(Z.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            temp_x = np.array([X[j], Y[i]])
            Z[i,j] = GaussMixture(temp_x, mean_list, cov_list,10)



    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X_,Y_, Z, alpha=0.5,cmap=cm.coolwarm)


    start = np.array([0,0])
    goal = np.array([8,4])

    ax.scatter(start[0], start[1], GaussMixture(start, mean_list, cov_list,10), marker='*')
    ax.scatter(goal[0], goal[1], GaussMixture(goal, mean_list, cov_list,10), marker='*')
    
    print(get_dist(start, goal, mean_list, cov_list,10))
    print(np.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2))

    plt.show()


