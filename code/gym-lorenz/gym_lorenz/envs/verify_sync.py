import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import pandas as pd
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
        # 或者简化为误差系统 [ex, ey, ez]，这里我们用全状态，让 Agent 自己学习特征
        self.observation_space = spaces.Box(low=-50.0, high=50.0, shape=(3,), dtype=np.float32)
        
        # HR 模型参数 (根据论文调整)
        self.a, self.b, self.c, self.d = 1.0, 3.0, 1.0, 5.0
        self.r, self.s, self.I_bias, self.x_rest = 0.006, 4.0, 3.2, -1.6
        self.dt = 0.001  # 仿真步长
        self.sigma=0.0 # 噪声强度，默认不添加噪声
        
        self.state_master = None # [x1, x2, x3]
        self.state_slave = None  # [y1, y2, y3]
        # --- 新增：步数计数器 ---
        self.current_step = 0
        self.max_steps = 2000  # 对应 2秒 的仿真
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 随机初始化主从系统的状态，增加训练难度（泛化性）
        self.current_step = 0 # 重置步数
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
        return error_vector.astype(np.float32), {}

    def step(self, action):
        self.current_step += 1
        
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
        # 3. 计算奖励 (Reward Design - 论文的核心)
        # 目标：误差越小奖励越高，控制量越小奖励越高
        # --- 论文公式 (14): 奖励函数 (曼哈顿距离的负值) ---
        # rt = -(|x1-y1| + |x2-y2| + |x3-y3|)
        reward = -np.sum(np.abs(error_vector))- 0.001 * np.sum(np.square(action))
        
        # 4. 判断结束条件
        # 通常混沌同步训练会跑固定的步数，或者误差过大时强制停止
        terminated = False
        truncated = self.current_step >= self.max_steps  # 2000步后强制结束并重置
        
        return error_vector.astype(np.float32), float(reward), terminated, truncated, {}
def plot_synchronization_scenario(model_path, add_noise=False, sigma=2.0, title="Scenario"):
    """
    通用绘图函数
    :param model_path: 模型路径
    :param add_noise: 是否开启噪声
    :param sigma: 如果开启噪声，噪声强度是多少
    :param title: 图表大标题
    """
    # 1. 初始化环境并配置噪声
    env = HRSyncEnv(add_noise=add_noise)
    if add_noise:
        env.sigma = sigma  # 强制固定噪声强度，保证测试公平性
    
    # 2. 加载模型
    try:
        model = PPO.load(model_path, device='cpu')
    except Exception as e:
        print(f"跳过绘图: 无法加载模型 {model_path}")
        return

    # 3. 数据采集
    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    time_axis = np.linspace(0, 5, 5000) 

    for i in range(10): # 跑10条随机初始化的曲线
        obs, _ = env.reset()
        err_history, act_history = [], []

        for _ in range(5000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action)
            err_history.append(obs)
            act_history.append(action)
        
        errs, acts = np.array(err_history), np.array(act_history)

        # 分别绘制 X, Y, Z 误差和两个控制力
        for j in range(3):
            axes[j].plot(time_axis, errs[:, j], color=colors[i], alpha=0.6, linewidth=1)
        axes[3].plot(time_axis, acts[:, 0], color=colors[i], alpha=0.6, linewidth=1)
        axes[4].plot(time_axis, acts[:, 1], color=colors[i], alpha=0.6, linewidth=1)

    # 4. 样式美化 (统一坐标轴和标签)
    labels = ['Error $e_x$', 'Error $e_y$', 'Error $e_z$', 'Control $a_1$', 'Control $a_2$']
    for idx, ax in enumerate(axes):
        ax.set_ylabel(labels[idx])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', linewidth=1.2)
        # 统一 Y 轴量程（可选），方便直接对比 B 和 C 的抖动幅度
        if idx < 3: ax.set_ylim(-25, 25) 
        else: ax.set_ylim(-2, 2)

    axes[4].set_xlabel("Time (seconds)")
    plt.suptitle(f"Numerical Simulation: {title}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 自动保存图片，方便写论文
    file_name = title.replace(" ", "_").lower() + ".png"
    plt.savefig(file_name, dpi=300)
    print(f"图表已保存为: {file_name}")
    plt.show()
def evaluate_scenario(model_path, add_noise=False, num_episodes=30, steps_per_episode=5000):
    """
    运行多个回合，收集误差并计算 MAE 和 RMSE
    """
    env = HRSyncEnv(add_noise=add_noise)
    # 强制在 CPU 上推理，速度快且无警告
    model = PPO.load(model_path, device='cpu')
    
    all_errors = []

    print(f"正在测试: 模型={model_path}, 环境噪声={add_noise}...")

    for i in range(num_episodes):
        obs, _ = env.reset()
        episode_errors = []
        for t in range(steps_per_episode):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action)
            if t > 1000: # 跳过前1秒的剧烈波动期
                episode_errors.append(obs)# obs 就是 error_vector
        all_errors.append(episode_errors)

    all_errors = np.array(all_errors) # (10, 4000, 3)
    
    # 计算 MAE (公式 16)
    mae = np.mean(np.abs(all_errors))
    # 计算 RMSE (公式 15)
    rmse = np.sqrt(np.mean(np.square(all_errors)))
    
    return mae, rmse

def main():
    # 这里填入你训练好的两个模型路径
    clean_model_path = "hr_paper_final_model.zip"      # 场景 A 训练的模型
    noisy_model_path = "hr_paper_noisy_model.zip"      # 场景 C 训练的模型（如果还没练，可以先填同一个看对比）

    results = []

    # 1. Testing without noise (用干净模型测干净环境)
    mae1, rmse1 = evaluate_scenario(clean_model_path, add_noise=False)
    results.append(["Testing without noise", mae1, rmse1])
    # --- Scenario A: 理想状态 ---
    plot_synchronization_scenario(clean_model_path, add_noise=False, title="Scenario A (Clean Model, No Noise)")

    # 2. Testing with noise (training without noise) (用干净模型测噪声环境)
    mae2, rmse2 = evaluate_scenario(clean_model_path, add_noise=True)
    results.append(["Testing with noise (training without noise)", mae2, rmse2])
    # --- Scenario B: 脆弱性测试 ---
    plot_synchronization_scenario(clean_model_path, add_noise=True, sigma=2.0, title="Scenario B (Clean Model with Noise)")

    # 3. Testing with noise (training with noise) (用噪声模型测噪声环境)
    # 注意：运行这一步前，你需要先开启 add_noise=True 训练一个新模型
    try:
        mae3, rmse3 = evaluate_scenario(noisy_model_path, add_noise=True)
        results.append(["Testing with noise (training with noise)", mae3, rmse3])
    except:
        results.append(["Testing with noise (training with noise)", "N/A (需先训练噪声模型)", "N/A"])
    
    # --- Scenario C: 鲁棒性展示 ---
    plot_synchronization_scenario(noisy_model_path, add_noise=True, sigma=2.0, title="Scenario C (Robust Model with Noise)")

    # 格式化成表格
    df = pd.DataFrame(results, columns=["Different cases", "MAE", "RMSE"])
    
    print("\n" + "="*50)
    print("复现 Table 1 结果预览")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    
    # 保存到 CSV 方便写论文
    df.to_csv("table_1_results.csv", index=False)
    print("数据已保存至 table_1_results.csv")

if __name__ == "__main__":
    main()
    