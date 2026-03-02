import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import torch

class PMSM_Sync_Env(gym.Env):
    "pmsm 电机同步环境"
    def __init__(self,alpha=0.5,add_noise=False):
        super().__init__()
        # 1. 设定系统物理参数 (基于文献设定)
        self.sigma=5.46
        self.gamma=20.0
        self.dt=0.001
        # 动作放缩：网络输出 [-1, 1]，实际控制力为 [-50, 50]
        self.input_min = -1.0
        self.input_max = 1.0
        # 目标状态（稳定点）
        self.state1=np.zeros(3)
        self.state2=np.zeros(3)
        
        self.alpha=alpha
        self.add_noise=add_noise
        # 2. 定义动作空间 (Action Space)
        # 智能体输出 2 个控制力 a1, a2，最大控制力为 50
        self.f_max=50
        self.action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # 3. 定义观测空间 (Observation Space)
        # 观测值为 6 维向量：[x1, x2, x3, dx1, dx2, dx3]
        self.observation_space=spaces.Box(low=-np.inf,high= np.inf,shape=(6,),dtype=np.float32)
        # 内部状态变量初始化
        self.state = None
        self.current_step = 0
        self.max_steps = 500  # 假设每个回合跑 2 秒，2/0.001 = 2000 步 (可根据训练需要调整)
    def _get_derivatives(self,state,action,noise=[0,0,0]):
        x1,x2,x3=state
        a1,a2=action
        # 严格按照文献中的动力学方程编写
        dx1=-x1+x2*x3+a1+noise[0]
        dx2=-x2-x1*x3+self.gamma*x3+a2+noise[1]
        dx3=self.sigma*(x2-x3)+noise[2]
        return np.array([dx1, dx2, dx3], dtype=np.float32)
    def reset(self,seed=None,options=None):
        """回合重置：初始化系统状态"""
        super().reset(seed=seed)
        self.current_step=0
        # 初始值从 (-30, 30) 范围内随机采样,下划线确保seed的可复现性
        self.state1=self.np_random.uniform(low=-30,high=30,size=(3,)).astype(np.float32)
        self.state2=self.np_random.uniform(low=-30,high=30,size=(3,)).astype(np.float32)
        # 初始动作设为 0，用于计算初始导数
        initial_action = np.array([0.0, 0.0], dtype=np.float32)
        derivatives1 = self._get_derivatives(self.state1, initial_action)
        derivatives2 = self._get_derivatives(self.state2, initial_action)

        error_state=self.state1-self.state2
        error_derivative=derivatives1-derivatives2
        # 拼接状态和导数作为第一次观测
        obs=np.concatenate([error_state,error_derivative])
        return obs,{}
    def step(self,action):
        """与环境交互：执行动作，计算状态演化、奖励并判断是否结束"""
        self.current_step+=1
        self.target_system_noise=self.np_random.normal(loc=0,scale=3, size=(3,))
        clip_action=np.clip(action,self.input_min,self.input_max)  # 确保动作在 [-1, 1] 范围内
        actual_action=action*self.f_max  # 将动作放缩到实际控制力范围 [-50, 50]
        # 动作截断，防止神经网络输出超出物理限制

        action=np.clip(action,-self.f_max,self.f_max)
        
        # --- 状态更新 (使用简单的欧拉法进行数值积分) ---
        derivatives1=self._get_derivatives(self.state1,[0,0])
        derivatives2=self._get_derivatives(self.state2,actual_action,
                                           self.target_system_noise if self.add_noise else [0,0,0])

        self.state1+=derivatives1*self.dt
        self.state2+=derivatives2*self.dt
        # 构造最新的观测向量 (误差状态 + 误差导数)
        new_derivatives1=self._get_derivatives(self.state1,[0,0])
        new_derivatives2=self._get_derivatives(self.state2,actual_action)
        
        error_state=self.state1-self.state2
        error_derivative=new_derivatives1-new_derivatives2

        obs = np.concatenate((error_state, error_derivative))
        # --- 计算奖励 (Reward) ---
        # 计算系统总误差 (曼哈顿距离) [cite: 250, 258]
        error_sum = np.sum(np.abs(error_state))
        # 改进的非线性误差敏感奖励函数 
        # 当误差接近 0 时，加上 error^alpha 会提供强烈的梯度信号
        reward=-error_sum-error_sum**self.alpha

        # --- 判断终止条件 ---
        terminated = False
        truncated = False
        
        # 如果系统发散到了不可接受的范围，给予严厉惩罚并结束回合 [cite: 497]
        if error_sum > 1000:  
            reward = -1000.0
            terminated = True
        
        # 如果达到了设定的最大步数，截断回合
        if self.current_step >= self.max_steps:
            truncated = True
        
        
        
        return obs, float(reward), terminated, truncated, {}
    def render(self):
        """用于可视化，这里我们只简单打印当前状态"""
        print(f"Step: {self.current_step}, State: {self.state}")