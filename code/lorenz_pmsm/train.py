import sys
from matplotlib.axes import Axes
from pathlib import Path

# 1. 获取当前文件的绝对路径
current_dir =Path(__file__).resolve()
# 2. .parent 是 lorenz_filter，.parent.parent 就是 code 目录
# 3. 直接用 / 符号拼接子目录名
gym_lorenz_path=current_dir.parent.parent / "gym-lorenz"

# 4. 转换回字符串并加入环境变量
sys.path.append(str(gym_lorenz_path))

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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env_utils import make_env

# 注意：关于 MlpPolicy 的导入已经删掉。
# 请在后续代码中直接使用字符串，如： model = DDPG("MlpPolicy", env)

class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # 1. 物理序列定义
        self.seq_len = 3      # 3 个物理轴 (d轴, q轴, w轴)
        self.raw_token_dim = 2 # 每个轴包含 2 个信息：(误差 e, 误差导数 de)
        # 注意力机制在维度太低时效果不好，所以我们把 2维 投射到 32维 的嵌入空间
        self.embed_dim=32
        self.token_embedding=nn.Linear(self.raw_token_dim,self.embed_dim)
        # 2. 定义自注意力层
        # batch_first=True 表示输入格式为 [batch, seq_len, embed_dim]
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            batch_first=True
        )
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        # 3. 后处理层：将注意力融合后的序列展平，映射到最终的 features_dim
        self.post_attention_fc=nn.Sequential(
            nn.Linear(self.seq_len*self.embed_dim, features_dim),
            nn.Tanh()# 遵循文献使用 Tanh
        )

    def forward(self, observations:torch.Tensor)->torch.Tensor:
        # observations shape: [batch_size, 6]
        batch_size=observations.shape[0]
        # --- 步骤 1：物理切分 ---
        # 假设输入顺序为 [e1, e2, e3, de1, de2, de3]
        # 我们需要把它重新组合成 [[e1, de1], [e2, de2], [e3, de3]]
        e_state = observations[:, 0:3]  # [batch, 3]
        de_state = observations[:, 3:6] # [batch, 3]
        # 用 stack 把它们在最后一个维度上拼起来，变成 [batch_size, 3, 2]
        x_seq = torch.stack([e_state, de_state], dim=-1)  # [batch, 3, 2]
        # 这样就完美切分出了 3 个物理 Token！
        # --- 步骤 2：特征升维 ---
        # 把二维物理量提升到 32 维的隐含空间 [batch_size, 3, 32]
        x_seq = self.token_embedding(x_seq)
        # --- 步骤 3：计算自注意力 ---
        # 3个轴开始互相交流，计算耦合关系
        attn_output,_ =self.attention_layer(x_seq,x_seq,x_seq)  # Self-Attention
        # 残差连接与归一化
        x_seq = self.layer_norm(x_seq + attn_output)
        # --- 步骤 4：展平与输出 ---
        # 展平成 [batch_size, 96]
        x_flattened = x_seq.reshape(batch_size,-1)
        # 映射到特征层 [batch_size, 64]
        features=self.post_attention_fc(x_flattened)
        return features
    
def train_pmsm_model(model_class,total_timesteps=50000,add_noise=False,alpha=None):
    # # 这里直接在 make 时传参，简化代码
    # env=make_env("PMSM_Sync_Env-v0",num_envs=8,add_noise=add_noise)
    # model=model_class("MlpPolicy",env,verbose=1,features_extractor_class=FeatureExtractor)
    # model.learn(total_timesteps=total_timesteps)
    # return model
    # 论文文献中要求的 8 个 alpha 值
    alpha_fractions = [(1, 2), (1, 3), (1, 4), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10)]
    # 1. 挂载我们之前写好的 PMSM 同步环境
    env=DummyVecEnv([lambda:Monitor(make_env(alpha=alpha,add_noise=add_noise))])
    # 2. 替换 VecFrameStack 为 VecNormalize
    # 自动归一化观测值，关闭奖励归一化以保留物理意义
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 3. 对齐文献的网络架构
    # policy_kwargs = dict(
    #     features_extractor_class=FeatureExtractor,
    #     features_extractor_kwargs=dict(features_dim=64),
    #     # 文献规定：隐藏层 64x64，激活函数 Tanh
    #     activation_fn=torch.nn.Tanh,
    #     net_arch=dict(pi=[64,64], vf=[64,64])
    # )

    # 4. 实例化模型，对齐文献超参数
    model=model_class(
        "MlpPolicy",
        env,
        learning_rate=0.001,  # 文献使用 0.001 的学习率
        # ent_coef=0.01,        
        verbose=1,
        n_steps=16,
        device='cpu',
        # policy_kwargs=policy_kwargs
        )
    
    noise_str = "noisy" if add_noise else "clean"
    print(f"正在训练 PMSM 模型 ({noise_str} 模式)，引入了自注意力机制，请稍候...")
    # 文献总训练步数为 500,000 [cite: 271]
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型和配套的归一化参数
    model.save(f"pmsm_attention_{noise_str}_model")
    env.save(f"pmsm_attention_{noise_str}_vecnorm.pkl")
    print(f"训练成功！模型已保存为 pmsm_attention_{noise_str}_model.zip")

def train_pure_pmsm_batch():
    alpha_fractions = [(1, 2), (1, 3), (1, 4), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10)]
    
    # 纯净版 MLP 架构
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh, # 或者用你发现的 ReLU 也可以
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    for num, den in alpha_fractions:
        alpha_val = num / den
        model_name = f"pmsm_a2c_alpha_{alpha_val:.2f}_clean_model"
        vecnorm_name = f"pmsm_a2c_alpha_{alpha_val:.2f}_clean_vecnorm.pkl" # 记得加上这个
        
        # 1. 挂载基础环境 (确保 dt=0.001 已经改好)
        env = DummyVecEnv([lambda: Monitor(make_env(alpha=alpha_val, add_noise=False))])
        
        # 2. 【核心加回】：套上 VecNormalize！
        # 极度注意：norm_reward=False 绝对不能改！我们要保留那个 -1000 的真实惩罚！
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        
        # 3. 实例化模型 (纯 MLP + 0.001 学习率)
        model = A2C("MlpPolicy",
                    env, 
                    learning_rate=linear_schedule(0.0005),
                    n_steps=16, 
                    max_grad_norm=0.5,                 # 【终极保险丝】：强行砍掉由于分数阶引发的爆炸梯度！
                    policy_kwargs=policy_kwargs, verbose=1, device='cpu')
        
        # 4. 训练
        model.learn(total_timesteps=1000_000)
        
        # 5. 【极其重要】：保存模型的同时，务必保存 VecNormalize 的统计数据！
        model.save(model_name)
        env.save(vecnorm_name)
from typing import Callable
def linear_schedule(initial_value:float)->Callable[[float], float]:
    """
    线性学习率调度器工厂函数。
    输入初始学习率，返回一个函数，该函数接受当前进度（0.0 到 1.0）并输出调整后的学习率。
    """
    def schedule(progress_remaining:float)->float:
        # progress_remaining 从 1.0 降到 0.0
        # 我们用它乘以初始学习率，实现线性递减
        # 为了防止最后期学习率变成绝对的 0 导致网络“脑死”，我们加一个极其微小的保底值 (比如 1e-5)
        return max(progress_remaining * initial_value, 1e-5)
    return schedule
if __name__ == "__main__":
    # 使用文献主推的 A2C 算法进行训练
    # train_pmsm_model(A2C, total_timesteps=500000, add_noise=False,alpha=0.1)
    # train_pmsm_model(A2C, total_timesteps=500000, add_noise=True,alpha=0.1)
    train_pure_pmsm_batch()