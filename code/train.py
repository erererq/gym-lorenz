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
        x_flattened = attn_output.reshape(-1, self.hidden_dim)  # [batch_size, hidden_dim]
        # 第五步：输出最终特征
        features = self.post_attention_fc(x_flattened)  # [batch_size, features_dim]
        return features


def train_model(model_class, total_timesteps=1000000,add_noise=False):
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
        model.save("hr_paper_noisy_model_smooth") # 这会生成一个 .zip 文件
    else:
        model.save("hr_paper_final_model_smooth") # 这会生成一个 .zip 文件
    print("训练成功！模型已保存为 zip")

if __name__ == "__main__":
    train_model(PPO, total_timesteps=500000, add_noise=False)  # 训练无噪声版本
    train_model(PPO, total_timesteps=500000, add_noise=True)  # 训练有噪声版本