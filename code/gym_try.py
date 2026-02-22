import os
import sys
from matplotlib.axes import Axes

# 1. 获取当前代码文件 (main_script.py) 所在的绝对路径目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 拼凑出 gym-lorenz 的绝对路径
gym_lorenz_path = os.path.join(current_dir, 'gym-lorenz')

# 3. 把拼出来的路径插到环境变量里
sys.path.append(gym_lorenz_path)

# 现在放心导入吧！

import time

# ==========================================
# 1. 基础科学计算与数据处理库
# ==========================================
import numpy as np
import pandas as pd
import networkx as nx
import math

# ==========================================
# 2. 深度学习底层库
# ==========================================
import torch 
import torch.nn as nn

# ==========================================
# 3. 可视化库 (Matplotlib)
# ==========================================
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter

# ==========================================
# 4. 强化学习相关库 (Gym & Stable Baselines3)
# ==========================================
import gym_lorenz  # 自定义环境
import gymnasium  # 最新版本的 Gym
from stable_baselines3 import DDPG, A2C, SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# 注意：关于 MlpPolicy 的导入已经删掉。
# 请在后续代码中直接使用字符串，如： model = DDPG("MlpPolicy", env)

class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    """
    改进版的自注意力特征提取器：将隐藏特征切分为序列，使注意力机制真正生效。
    """
    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        # 1. 基础特征提取
        # 假设我们将输入映射到一个可以被整除的隐藏维度，比如 128
        
        self.hidden_dim = 128
        self.fc1=nn.Linear(observation_space.shape[0], self.hidden_dim)  # 从输入维度到隐藏维度

        # 【关键改进点】: 定义序列切分规则
        # 我们把 128 维的特征，切分成 8 个 "Token"，每个 "Token" 维度是 16
        self.seq_len = 8
        self.token_dim = 16

        # 2. 定义一个自注意力层
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=self.token_dim, num_heads=4, batch_first=True)
        # 3. 后处理层：将注意力融合后的序列展平，映射到最终的 features_dim
        self.post_attention_fc=nn.Sequential(
            nn.Linear(self.hidden_dim, self.features_dim),
            nn.ReLU())  # 从隐藏维度到最终特征维度
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        :param observations: (batch_size, obs_dim)
        :return: (batch_size, features_dim)
        """
        # 1. 基础特征提取
        x = torch.relu(self.fc1(observations))  # [batch_size, hidden_dim]
        # 第二步：重塑为序列 (Reshape)
        # 将 [batch_size, 128] 变成 [batch_size, 8, 16]
        # 这样就有 8 个不同的元素可以互相计算注意力了！
        x_seq=x.view(-1,self.seq_len,self.token_dim)  # [batch_size, seq_len, token_dim]
        # 3. 自注意力层
        # 这时，8个特征块之间会产生动态的权重分配，理论上可以捕捉到输入特征之间更复杂的关系
        attn_output, _ = self.attention_layer(x_seq, x_seq, x_seq)  # Self-Attention
        # 第四步：将序列重新展平回 [batch_size, 128]
        x_flattened = attn_output.view(-1, self.hidden_dim)  # [batch_size, hidden_dim]
        # 第五步：输出最终特征
        features = self.post_attention_fc(x_flattened)  # [batch_size, features_dim]
        return features


def train_model(model_class, env, total_timesteps=1000000,add_noise=False):
    env_fn=lambda:gymnasium.make("lorenz_try-v0",add_noise=add_noise)  # 这里直接在 make 时传参，简化代码

    env=DummyVecEnv([env_fn])
    # from stable_baselines3.common.env_util import make_vec_env
    # # 一行代码搞定，自动使用 DummyVecEnv，如果要多线程只要修改 n_envs=4
    # env = make_vec_env('lorenz_transient-v0', n_envs=1)

    # 1. 合并 policy_kwargs，避免覆盖
    policy_kwargs = dict(
        features_extractor_class=AttentionFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=64),
        # 注意：在较新的 stable-baselines3 版本中，net_arch 的字典不需要包在 list 里
        net_arch=dict(pi=[128, 128], vf=[128, 128]) 
    )
    model = model_class("MlpPolicy", env, learning_rate=3e-4,      # 降低学习率，走稳一点
    n_steps=2048,            # 每次更新采集更多样本
    batch_size=64,           # 减小 Batch 增加更新频率
    gae_lambda=0.95,         # 稳定优势估计
    verbose=1,
    tensorboard_log="./lorenztensorboard1/",  # 添加 TensorBoard 日志路径
    device='cpu')
    print("正在训练论文版模型，请稍候...")
    model.learn(total_timesteps=total_timesteps)
    if add_noise:
        model.save("hr_paper_noisy_model") # 这会生成一个 .zip 文件
    else:
        model.save("hr_paper_final_model") # 这会生成一个 .zip 文件
    print("训练成功！模型已保存为 zip")

def plot_paper_style(tests_data,title,save_path):
    """根据传入的数据，画出符合论文风格的 5 联图"""
    plt.rcParams["font.family"]="serif"
    plt.rcParams["axes.facecolor"]="#D3D3D5"
    plt.rcParams["axes.edgecolor"]="black"
    plt.rcParams['axes.grid'] = False

    fig,axs=plt.subplots(5,1,figsize=(8,20))

    titles = ["Error in X", "Error in Y", "Error in Z", "Control Force of X", "Control Force of Y"]
    ylabels = ["Error", "Error", "Error", "control term", "control term"]

    for ax,ax_title ,ylabel in zip(axs, titles, ylabels):
        ax.set_title(ax_title, fontsize=14)
        ax.set_xlabel("Time(s)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
    # 遍历 10 次测试的数据，画出 10 条线
    for i,data in enumerate(tests_data):
        test_id=i+1
        t=data["time"]

        axs[0].plot(t, data['err_x'], label=f"Test {test_id}(initial Error={data['init_err_x']:.1f})", linewidth=1)
        axs[1].plot(t, data['err_y'], label=f"Test {test_id}(initial Error={data['init_err_y']:.1f})", linewidth=1)
        axs[2].plot(t, data['err_z'], label=f"Test {test_id}(initial Error={data['init_err_z']:.1f})", linewidth=1)
        axs[3].plot(t, data['ctrl_x'], label=f"Test {test_id}(initial Control={data['init_ctrl_x']:.1f})", linewidth=1)
        axs[4].plot(t, data['ctrl_y'], label=f"Test {test_id}(initial Control={data['init_ctrl_y']:.1f})", linewidth=1)
    
    for ax in axs:
        ax:Axes
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8, edgecolor='white')
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存为: {save_path}")
    plt.close() # 画完关掉，防止内存溢出

# ==========================================
# 模块 2：测试与评估函数（负责跑环境、收集数据、算 MAE/RMSE）
# ==========================================
def test_and_evaluate(model_path, add_noise, title):
    print(f"\n▶ 正在测试场景: {title} (模型: {model_path})")

    # 1. 初始化环境
    env = gymnasium.make("lorenz_try-v0",add_noise=add_noise)
    if add_noise:
        env.sigma = 2.0  # 锁定噪声强度

    # 2. 尝试加载模型
    try:
        model = PPO.load(model_path,env, device='cpu')
    except Exception as e:
        print(f"⚠️ 无法加载模型 {model_path}，跳过此场景。")
        return "N/A", "N/A"  # 如果找不到模型，直接返回 N/A

    all_tests_data = [] # 用来给画图函数用的
    steady_errors = []  # 用来算 MAE 和 RMSE 的 (去掉前1000步)

    # 3. 跑 10 次独立测试
    for test_idx in range(10):
        obs,_=env.step()

        # 准备一个小本子，记录当前这 1 次测试的所有过程
        test_record = {'time': [], 'err_x': [], 'err_y': [], 'err_z': [], 'ctrl_x': [], 'ctrl_y': []}
        test_record['init_err_x'] = obs[0]
        test_record['init_err_y'] = obs[1]
        test_record['init_err_z'] = obs[2]
        first_action_recorded = False

        # 跑 5000 步
        for step in range(5000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action)
            # 记录测试数据
            test_record['time'].append(step*0.001)  # 记录时间，单位秒
            test_record['err_x'].append(obs[0]*50)  # 乘回去，记录真实误差
            test_record['err_y'].append(obs[1]*50)  # 乘回去，记录真实误差
            test_record['err_z'].append(obs[2]*50)  # 乘回去，记录真实误差
            test_record['ctrl_x'].append(action[0]*100)
            test_record['ctrl_y'].append(action[1]*100)

            if not first_action_recorded:
                test_record['init_ctrl_x'] = action[0]
                test_record['init_ctrl_y'] = action[1]
                first_action_recorded = True

            # 如果过了前 1000 步（1秒），系统稳定了，就把误差存起来算 MAE
            if step >= 1000:
                steady_errors.append(obs)
        
        all_tests_data.append(test_record)
    
    # 4. 计算最终的 MAE 和 RMSE
    steady_errors = np.array(steady_errors)
    mae = np.mean(np.abs(steady_errors))
    rmse = np.sqrt(np.mean(np.square(steady_errors)))
    # 5. 调用上面的画图函数出图
    save_name = title.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "") + ".png"
    plot_paper_style(all_tests_data, title=title, save_path=save_name)

    return mae, rmse

# ==========================================
# 模块 3：主控程序（负责安排三个场景，并打印表格）
# ==========================================
def main():
    # 填入你的模型路径
    clean_model_path = "hr_paper_final_model.zip"      
    noisy_model_path = "hr_paper_noisy_model.zip"      

    results = []

    # 场景 A: 干净模型 测 干净环境
    mae1, rmse1 = test_and_evaluate(clean_model_path, add_noise=False, title="Scenario A (Clean Model, No Noise)")
    results.append(["Testing without noise", mae1, rmse1])

    # 场景 B: 干净模型 测 噪声环境 (脆弱性测试)
    mae2, rmse2 = test_and_evaluate(clean_model_path, add_noise=True, title="Scenario B (Clean Model with Noise)")
    results.append(["Testing with noise (training without noise)", mae2, rmse2])

    # 场景 C: 噪声模型 测 噪声环境 (鲁棒性展示)
    mae3, rmse3 = test_and_evaluate(noisy_model_path, add_noise=True, title="Scenario C (Robust Model with Noise)")
    results.append(["Testing with noise (training with noise)", mae3, rmse3])

    # --- 打印和保存最终的对比表格 ---
    df = pd.DataFrame(results, columns=["Different cases", "MAE", "RMSE"])
    
    print("\n" + "="*60)
    print("复现论文 Table 1 结果预览")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    df.to_csv("table_1_results.csv", index=False)
    print("✅ 数据已保存至 table_1_results.csv")

if __name__ == "__main__":
    main()