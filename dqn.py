import numpy as np
import torch
from torch.autograd import Variable
import copy
from random import random
from gym import Env
import gym
from weatherenv2 import *
from core import Transition, Experience, Agent

class Approximator(torch.nn.Module):
    '''base class of different function approximator subclasses
    '''
    def __init__(self, dim_input=1, dim_output=1, dim_hidden=16):
        super(Approximator, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden

        self.linear1 = torch.nn.Linear(self.dim_input, self.dim_hidden)
        self.linear2 = torch.nn.Linear(self.dim_hidden, self.dim_output)

    def _forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)  # relu
        y_pred = self.linear2(h_relu)
        return y_pred

    def fit(self, x, y, criterion=None, optimizer=None, epochs=1, learning_rate=1e-4):
        if criterion is None:
            criterion = torch.nn.MSELoss(size_average=False)
        
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if epochs < 1:
            epochs = 1

        x = self._prepare_data(x)
        y = self._prepare_data(y, False)

        for _ in range(epochs):
            y_pred = self._forward(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss

    def _prepare_data(self, x, requires_grad=True):
        
        if isinstance(x, np.ndarray):
            x = x.astype(np.float64)
            x = Variable(torch.from_numpy(x), requires_grad=requires_grad)
        if isinstance(x, int):
            x = Variable(torch.Tensor([[x]]), requires_grad=requires_grad)

        x = x.float()  # 从from_numpy()转换过来的数据是DoubleTensor形式
        # x = x.astype(np.float64)
        if x.data.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def __call__(self, x):
        x = self._prepare_data(x)
        pred = self._forward(x)
        return pred.data.numpy()

    def clone(self):
        return copy.deepcopy(self)

class ApproxQAgent(Agent):
    def __init__(self, env: Env = None, trans_capacity=20000, hidden_dim: int = 16):
        if env is None:
            raise "agent should have an environment"
        super(ApproxQAgent, self).__init__(env, trans_capacity)
        self.input_dim, self.output_dim = 1, 1

        # 适应不同的状态和行为空间类型
        if isinstance(env.observation_space, spaces.Discrete):
            self.input_dim = 1
        elif isinstance(env.observation_space, spaces.Box):
            self.input_dim = env.observation_space.shape[0]

        if isinstance(env.action_space, spaces.Discrete):
            self.output_dim = env.action_space.n
        elif isinstance(env.action_space, spaces.Box):
            self.output_dim = env.action_space.shape[0]

        self.hidden_dim = hidden_dim
        self.Q = Approximator(dim_input = self.input_dim,
                              dim_output = self.output_dim,
                              dim_hidden = self.hidden_dim)
        self.PQ = self.Q.clone()

        self.plot_steps = []
        self.plot_rewards = []
        self.plot_loss = []
        self.plot_epsilon = []


    def _learn_from_memory(self, gamma, batch_size, learning_rate, epochs):
        trans_pieces = self.sample(batch_size)  # 随机获取记忆里的Transmition
        states_0 = np.vstack([x.s0 for x in trans_pieces])
        actions_0 = np.array([x.a0 for x in trans_pieces])
        reward_1 = np.array([x.reward for x in trans_pieces])
        is_done = np.array([x.is_done for x in trans_pieces])
        states_1 = np.vstack([x.s1 for x in trans_pieces])

        X_batch = states_0
        y_batch = self.Q(states_0)  # 得到numpy格式的结果

        # 使用了Batch，代码是矩阵运算
        Q_target = reward_1 + gamma * np.max(self.Q(states_1), axis=1)*\
            (~ is_done) # is_done则Q_target==reward_1
        y_batch[np.arange(len(X_batch)), actions_0] = Q_target
        # loss is a torch Variable with size of 1
        loss = self.PQ.fit(x = X_batch, 
                           y = y_batch, 
                           learning_rate = learning_rate,
                           epochs = epochs)
        mean_loss = loss.sum().item() / batch_size
        self._update_Q_net()
        return mean_loss

    def learning(self, gamma = 0.99,
                       learning_rate=1e-5, 
                       max_episodes=1000, 
                       batch_size = 64,
                       min_epsilon = 0.2,
                       epsilon_factor = 0.1,
                       epochs = 1):
        """learning的主要工作是构建经历, 当构建的经历足够时, 同时启动基于经历的学习
        """
        total_steps, step_in_episode, num_episode = 0, 0, 0
        target_episode = max_episodes * epsilon_factor
        while num_episode < max_episodes:
            epsilon = self._decayed_epsilon(cur_episode = num_episode,
                                            min_epsilon = min_epsilon, 
                                            max_epsilon = 1,
                                            target_episode = target_episode)
            self.state = self.env.reset()
            self.env.render()
            step_in_episode = 0
            loss, mean_loss = 0.00, 0.00
            is_done = False
            while not is_done:
                s0 = self.state
                a0  = self.performPolicy(s0, epsilon)
                # act方法封装了将Transition记录至Experience中的过程
                s1, r1, is_done, info, total_reward = self.act(a0)
                # self.env.render()
                step_in_episode += 1
                # 当经历里有足够大小的Transition时，开始启用基于经历的学习
                if self.total_trans > batch_size:
                    loss += self._learn_from_memory(gamma, 
                                                    batch_size, 
                                                    learning_rate,
                                                    epochs)
            mean_loss = loss / step_in_episode
            print("{0} epsilon:{1:3.2f}, loss:{2:.3f}".
                format(self.experience.last, epsilon, mean_loss))
            self.plot_steps.append(len(self.experience.last.trans_list))
            self.plot_epsilon.append(epsilon)
            self.plot_rewards.append(self.experience.last.total_reward)
            self.plot_loss.append(mean_loss)
            # print(self.experience)
            total_steps += step_in_episode
            num_episode += 1
        return  

    def _decayed_epsilon(self,cur_episode: int, 
                              min_epsilon: float, 
                              max_epsilon: float, 
                              target_episode: int) -> float:
        '''获得一个在一定范围内的epsilon
        '''
        slope = (min_epsilon - max_epsilon) / (target_episode)
        intercept = max_epsilon
        return max(min_epsilon, slope * cur_episode + intercept)

    def _curPolicy(self, s, epsilon = None):
        '''依据更新策略的价值函数(网络)产生一个行为
        '''
        Q_s = self.PQ(s)
        rand_value = random()
        if epsilon is not None and rand_value < epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(Q_s))

    def performPolicy(self, s, epsilon = None):
        return self._curPolicy(s, epsilon)

    def _update_Q_net(self):
        '''将更新策略的Q网络(连带其参数)复制给输出目标Q值的网络
        '''
        self.Q = self.PQ.clone()

    def showFinalRoute(self):
        total_steps = 0
        epsilon = None
        self.state = self.env.reset()
        self.env.render_final()
        step_in_episode = 0
        loss, mean_loss = 0.00, 0.00
        is_done = False
        while not is_done:
            s0 = self.state
            a0  = self.performPolicy(s0, epsilon)
            # act方法封装了将Transition记录至Experience中的过程
            s1, r1, is_done, info, total_reward = self.act(a0)
            self.env.render_final()
            step_in_episode += 1
        print(self.experience.last)
        total_steps += step_in_episode
        return

    def plot(self):
        import matplotlib.pyplot as plt
        x = list(range(len(self.plot_steps)))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        ax1.set_title('total steps')
        ax2.set_title('total rewards')
        ax3.set_title('epsilon')
        ax4.set_title('loss')

        ax1.plot(x, self.plot_steps, 'b')
        ax2.plot(x, self.plot_rewards, 'b')
        ax3.plot(x, self.plot_epsilon, 'r')
        ax4.plot(x, self.plot_loss, 'r')

        fig.savefig('curves.png')



def testApproxQAgent():
    env = WeatherWorld()
    # env = gym.make("CartPole-v1")
    
    # directory = "/Users/perry/miniforge3/envs/rl/lib/python3.8/site-packages/gym/wrappers/monitor"

    # env = gym.wrappers.Monitor(env, directory, force=True)
    agent = ApproxQAgent(env,
                         trans_capacity = 10000,    # 记忆容量（按状态转换数计）
                         hidden_dim = 16)           # 隐藏神经元数量
    env.reset()
    print("Learning...")  
    agent.learning(gamma=0.99,          # 衰减引子
                   learning_rate = 1e-3,# 学习率
                   batch_size = 64,     # 集中学习的规模
                   max_episodes=2000,   # 最大训练Episode数量
                   min_epsilon = 0.01,   # 最小Epsilon
                   epsilon_factor = 0.3,# 开始使用最小Epsilon时Episode的序号占最大
                                        # Episodes序号之比，该比值越小，表示使用
                                        # min_epsilon的episode越多
                   epochs = 2           # 每个batch_size训练的次数
                   )
    agent.showFinalRoute()
    input("press any key to continue...")

    agent.plot()

    





if __name__ == "__main__":
    testApproxQAgent()