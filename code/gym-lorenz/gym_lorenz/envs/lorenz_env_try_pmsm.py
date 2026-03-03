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

        self.lambda_coef = 0.0  # 初始惩罚系数为 0，完全不限制力矩
        # =======================================================
        # 🚀 高级机制：为 Lambda 专属定制的 Adam 优化器参数
        # =======================================================
        self.lambda_lr = 0.001   # Adam 的基础学习率（因为有自适应，可以给大一点）
        self.beta1 = 0.9         # 一阶动量衰减系数 (记录方向的惯性)
        self.beta2 = 0.999       # 二阶动量衰减系数 (记录震荡的方差)
        self.epsilon = 1e-8      # 防止除以 0 的极小值
        # Adam 的内部记忆状态
        self.m_t = 0.0           # 一阶动量 (Momentum)
        self.v_t = 0.0           # 二阶动量 (Variance)
        self.adam_step = 0       # 独立的 Adam 步数计数器
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
        # --- 新增：全局 Lambda 衰减时钟 ---
        self.global_step = 0
        self.total_training_steps = 1_000_000  # 替换掉冲突的 max_steps，代表整个训练总步数
        self.initial_lambda_lr = 0.0001        # lambda 的初始更新步伐
        self.current_step = 0
        self.max_steps = 2000  # 假设每个回合跑 2 秒，2/0.001 = 2000 步 (可根据训练需要调整)
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
        # self.global_step += 1  # 每次 step，全局时钟也往前走
        self.target_system_noise=self.np_random.normal(loc=0,scale=3, size=(3,))
        clip_action=np.clip(action,self.input_min,self.input_max)  # 确保动作在 [-1, 1] 范围内
        actual_action=clip_action*self.f_max  # 将动作放缩到实际控制力范围 [-50, 50]
        # # 动作截断，防止神经网络输出超出物理限制

        # action=np.clip(action,-self.f_max,self.f_max)
        
        # --- 状态更新 (使用简单的欧拉法进行数值积分) ---
        derivatives1=self._get_derivatives(self.state1,[0,0])
        derivatives2=self._get_derivatives(self.state2,actual_action,
                                           self.target_system_noise if self.add_noise else [0,0,0])

        self.state1+=derivatives1*self.dt
        self.state2+=derivatives2*self.dt
        # 构造最新的观测向量 (误差状态 + 误差导数)
        new_derivatives1=self._get_derivatives(self.state1,[0,0])
        new_derivatives2=self._get_derivatives(self.state2,actual_action,
                                               self.target_system_noise if self.add_noise else [0,0,0])
        
        error_state=self.state1-self.state2
        error_derivative=new_derivatives1-new_derivatives2

        obs = np.concatenate((error_state, error_derivative))
        # --- 计算奖励 (Reward) ---
        # 计算系统总误差 (曼哈顿距离) [cite: 250, 258]
        e1=np.abs(error_state[0])
        e2=np.abs(error_state[1])
        e3=np.abs(error_state[2])
        error_sum = np.sum([e1, e2, e3])

        # =======================================================
        # 🧠 核心数学引擎：Adam-based Dual Descent (对偶梯度下降)
        # =======================================================
        error_threshold = 5.0 
        
        # 1. 计算损失函数的“梯度” (Gradient of the Constraint)
        # 数学推导：我们希望 error_sum > threshold 时，lambda 增加。
        # 因此，针对 lambda 的下降梯度必须是负数。
        grad = error_threshold - error_sum 
        
        # 2. 更新 Adam 的“心跳”
        self.adam_step += 1
        
        # 3. 计算一阶动量 (梯度的指数移动平均) -> 让财务部不要听风就是雨，要看长期趋势！
        self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * grad
        
        # 4. 计算二阶动量 (梯度的平方的指数移动平均) -> 记录混沌系统震荡的剧烈程度！
        self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * (grad ** 2)
        
        # 5. 偏差校正 (Bias Correction，消除初期的零初始化误差)
        m_hat = self.m_t / (1 - self.beta1 ** self.adam_step)
        v_hat = self.v_t / (1 - self.beta2 ** self.adam_step)
        
        # 6. 自适应步长更新 Lambda
        # 神奇之处：除以 sqrt(v_hat) 意味着，如果系统在剧烈震荡，更新步幅会自动缩小，绝对不会崩溃！
        self.lambda_coef -= self.lambda_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # 7. 对偶变量投影 (Projected Gradient) -> 保证财务部不会贴钱 (>=0)，也不会暴走 (<=0.5)
        self.lambda_coef = np.clip(self.lambda_coef, 0.0, 0.5)
        
        # =======================================================
        # 改进的非线性误差敏感奖励函数 
        # # --- 核心动态博弈机制 ---
        # error_threshold = 5.0 # 我们能容忍的最大物理误差

        # # 动态计算当前 lambda 的更新步长
        # progress_remaining = 1.0 - (self.global_step / self.total_training_steps)
        # current_lambda_lr = self.initial_lambda_lr * progress_remaining
        # current_lambda_lr = max(current_lambda_lr, 1e-6) # 保证步长不能低于 1e-6
        
        # if error_sum < error_threshold:
        #     # 表现很好，误差在 5.0 以内。开始收紧预算，逼它平滑！
        #     self.lambda_coef += current_lambda_lr 
        # else:
        #     # 表现很差，误差失控了。赶紧放宽预算，允许它输出大电流去救场！
        #     self.lambda_coef -= current_lambda_lr
        # 分数阶误差惩罚 (极度严苛的逼近)
        # 加上 1e-6 是为了防止底数为 0 时，在某些极端情况下的反向传播计算出现 NaN
        fractional_penalty = (abs(e1) + 1e-6)**self.alpha + \
                             (abs(e2) + 1e-6)**self.alpha + \
                             (abs(e3) + 1e-6)**self.alpha
        # 动作惩罚 (极其重要！用来镇压分数阶梯度带来的狂暴震荡)
        # # 防止系数变成负数或者大得离谱
        # self.lambda_coef = np.clip(self.lambda_coef, 0.0, 0.5)
        
        action_penalty = self.lambda_coef * (action[0]**2 + action[1]**2)
        # 当误差接近 0 时，加上 error^alpha 会提供强烈的梯度信号
        reward=-error_sum-fractional_penalty-action_penalty

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