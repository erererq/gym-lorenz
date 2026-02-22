import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import torch

def hr_derivatives(state, a1, a2, a, b, c, d, r, s, I_bias, x_rest):
    x1, x2, x3 = state
    dx1 = x2 - a*(x1**3) + b*(x1**2) - x3 + I_bias
    dx2 = c - d*(x1**2) - x2 + a1
    dx3 = r * (s * (x1 - x_rest) - x3) + a2
    return np.array([dx1, dx2, dx3])
class HRSyncEnv(gym.Env):
    """
    Hindmarsh-Rose 神经元同步环境
    Master 系统: 自由运行
    Slave 系统: 受 RL Agent 控制
    """
    def __init__(self, add_noise=False):
        super().__init__()
        self.add_noise = add_noise  # 开关：是否添加噪声
        
        # 1. 定义动作空间: 连续值，代表控制电流 u
        # 假设控制电流范围在 [-1.0, 1.0] 之间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 2. 定义状态空间: [x_m, y_m, z_m, x_s, y_s, z_s]
        # 或者简化为误差系统 [ex, ey, ez]，这里我们用误差，让 Agent 自己学习特征
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.scale_factor = 50.0  # 定义缩放因子
        # HR 模型参数 (根据论文调整)
        self.a, self.b, self.c, self.d = 1.0, 3.0, 1.0, 5.0
        self.r, self.s, self.I_bias, self.x_rest = 0.006, 4.0, 3.2, -1.6
        self.dt = 0.001  # 仿真步长
        self.sigma=0.0 # 噪声强度，默认不添加噪声
        
        self.state_master = None # [x1, x2, x3]
        self.state_slave = None  # [y1, y2, y3]
        # --- 新增：步数计数器 ---
        # self.current_step = 0
        # self.max_steps = 2000  # 对应 2秒 的仿真
        # 限制最大步数，防止发散过久，这点在注册时已经设置 max_episode_steps=2000，
        # 所以就不需要在这里重复设置了，下面同理
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 随机初始化主从系统的状态，增加训练难度（泛化性）
        # self.current_step = 0 # 重置步数
        # Master 系统初始值
        # 按照论文：在 [-10, 20] 之间随机初始化
        self.state_master = np.random.uniform(-10, 20, 3)
        # Slave 系统初始值（给它一个较大的初始偏差）
        self.state_slave = np.random.uniform(-10, 20, 3)
        if self.add_noise:
            # 每次重置环境，噪声强度都变一下，范围 [0, 2]
            self.sigma = np.random.uniform(0, 2)
        else:
            self.sigma = 0.0
        # 返回论文要求的观察向量 Ot (公式 12)
        error_vector = self.state_master - self.state_slave
        # 记得 Reset 也要归一化
        normalized_error = np.clip(error_vector / self.scale_factor, -1.0, 1.0)
        return normalized_error.astype(np.float32), {}

    def step(self, action):
        # self.current_step += 1
        
        # 映射动作
        # --- 关键修改：手动将 [-1, 1] 映射到 [-100, 100] ---
        # 假设网络输出的是 raw_action
        a1 = np.clip(action[0], -1, 1) * 100.0
        a2 = np.clip(action[1], -1, 1) * 100.0

        # --- RK4 参数准备 ---
        dt = self.dt
        params = (self.a, self.b, self.c, self.d, self.r, self.s, self.I_bias, self.x_rest)

        # --- 更新 Master 系统 (注意 Master 不受 a1, a2 控制) ---
        m_s = self.state_master
        mk1 = hr_derivatives(m_s, 0, 0, *params)
        mk2 = hr_derivatives(m_s + dt/2 * mk1, 0, 0, *params)
        mk3 = hr_derivatives(m_s + dt/2 * mk2, 0, 0, *params)
        mk4 = hr_derivatives(m_s + dt * mk3, 0, 0, *params)
        self.state_master += (dt/6.0) * (mk1 + 2*mk2 + 2*mk3 + mk4)

        # --- 更新 Slave 系统 (受 a1, a2 控制) ---
        s_s = self.state_slave
        sk1 = hr_derivatives(s_s, a1, a2, *params)
        sk2 = hr_derivatives(s_s + dt/2 * sk1, a1, a2, *params)
        sk3 = hr_derivatives(s_s + dt/2 * sk2, a1, a2, *params)
        sk4 = hr_derivatives(s_s + dt * sk3, a1, a2, *params)
        self.state_slave += (dt/6.0) * (sk1 + 2*sk2 + 2*sk3 + sk4)

        
        # self.current_step += 1
        # x1, x2, x3 = self.state_master
        # y1, y2, y3 = self.state_slave
        # # --- 关键修改：手动将 [-1, 1] 映射到 [-100, 100] ---
        # # 假设网络输出的是 raw_action
        # a1 = np.clip(action[0], -1, 1) * 100.0 
        # a2 = np.clip(action[1], -1, 1) * 100.0

        # # --- 论文公式 (10): Drive System (Master) ---
        # dx1 = x2 - self.a*(x1**3) + self.b*(x1**2) - x3 + self.I_bias
        # dx2 = self.c - self.d*(x1**2) - x2
        # dx3 = self.r * (self.s * (x1 - self.x_rest) - x3)
        
        # # --- 论文公式 (13): Response System (Slave + Control) ---
        # dy1 = y2 - self.a*(y1**3) + self.b*(y1**2) - y3 + self.I_bias
        # dy2 = self.c - self.d*(y1**2) - y2 + a1  # 控制量 a1 加在这里
        # dy3 = self.r * (self.s * (y1 - self.x_rest) - y3) + a2 # 控制量 a2 加在这里
        # 3. 产生 3D 高斯噪声 (假设标准差 sigma 在 reset 中已采样)
        # 仅在训练或特定测试场景下开启
        # 只有在测试 Scenario B 或 C 时，才把这个开关打开
        if self.add_noise:
            noise = np.random.normal(0, self.sigma, 3)
            self.state_master += noise * self.dt  # 噪声随步长缩放
            # # 作用于 Master 系统
            # dx1 += noise[0]
            # dx2 += noise[1]
            # dx3 += noise[2]
        

        # # 更新数值积分 (Euler 方法)
        # self.state_master += np.array([dx1, dx2, dx3]) * self.dt
        # self.state_slave += np.array([dy1, dy2, dy3]) * self.dt
        # --- 后续逻辑不变 (计算 reward, observation 等) ---
        # 使用 RK4 或简单的欧拉法进行一阶段更新 (为了训练速度，这里示例用改进欧拉)
        # --- 论文公式 (12): 计算新的观察向量 Ot ---
        error_vector = self.state_master - self.state_slave
        normalized_error = error_vector / self.scale_factor
        # 3. 计算奖励 (Reward Design - 论文的核心)
        # 目标：误差越小奖励越高，控制量越小奖励越高
        # --- 论文公式 (14): 奖励函数 (曼哈顿距离的负值) ---
        # rt = -(|x1-y1| + |x2-y2| + |x3-y3|)
         # --- 改动 3: 奖励函数保持使用“真实误差” ---
        # 为什么要用真实误差？因为我们需要物理意义上的收敛。
        # 如果用归一化误差，Reward 数值太小，可能需要调整权重。
        # 这里的 0.001 * action^2 是惩罚 [-1,1] 的动作输出，是合理的
        reward = -np.sum(np.abs(error_vector))- 0.001 * np.sum(np.square(action))
        
        # 4. 判断结束条件
        # 通常混沌同步训练会跑固定的步数，或者误差过大时强制停止
        terminated = False
        truncated = False
        # --- 改动 4: 增加 Early Stopping (防发散) ---
        # 如果任何一个维度的误差超过 60 (比50大一点)，认为控制失败，强制结束
        # 这能极大地加速训练，不让 Agent 在错误的道路上浪费时间
        if np.any(np.abs(error_vector) > 60.0):
            terminated = True
            reward = -2000.0  # 给一个大的惩罚
        # truncated = self.current_step >= self.max_steps  # 2000步后强制结束并重置
        
        return normalized_error.astype(np.float32), float(reward), terminated, truncated, {}

if __name__ == "__main__":
    env = HRSyncEnv(add_noise=False)  # 启用噪声
    model = PPO("MlpPolicy", env, learning_rate=3e-4,      # 降低学习率，走稳一点
    n_steps=2048,            # 每次更新采集更多样本
    batch_size=64,           # 减小 Batch 增加更新频率
    gae_lambda=0.95,         # 稳定优势估计
    verbose=1,
    device='cpu')
    print("正在训练论文版模型，请稍候...")
        #  100 万步
    model.learn(total_timesteps=1000000)
    model.save("hr_paper_final_model") # 这会生成一个 .zip 文件
    print("训练成功！模型已保存为 hr_paper_final_model.zip")