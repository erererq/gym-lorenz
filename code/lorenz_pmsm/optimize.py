import gymnasium as gym
import numpy as np
import optuna
import torch 
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor # 记得导入

import sys
from pathlib import Path
current_dir=Path(__file__).resolve()

gym_lorenz_path=current_dir.parent.parent/"gym-lorenz"

sys.path.append(str(gym_lorenz_path))

import gym_lorenz
import gymnasium

def objective(trial):
    # --- A. 让 Optuna 自动采样我们关心的超参数 ---
    
    # 1. 算法超参数
    # 学习率：在 1e-5 到 1e-2 之间按对数尺度搜索
    lr=trial.suggest_float("learning_rate",1e-5,1e-2,log=True)
    # 熵系数：决定探索力度，对混沌系统极其重要
    # ent_coef=trial.suggest_float("ent_coef",0.00001,0.1,log=True)
    # 前瞻步数：A2C 的 n_steps (必须是2的幂次比较好)
    n_steps= trial.suggest_categorical("n_steps",[8,16,32,64,128])

    # 2. 物理环境超参数
    # 奖励函数中的 alpha：看看是不是文献说的 0.5 最好，我们在 0.1 到 1.0 之间搜
    # env_alpha=trial.suggest_float("alpha",0.1,1.0)

    # 3. 网络结构审美
    # 让 Optuna 决定用 [64, 64] 还是 [128, 128]
    net_dim= trial.suggest_categorical("net_dim",[64,128])
    policy_kwargs=dict(
        activation_fn=torch.nn.Tanh,  # 激活函数
        net_arch=dict(pi=[net_dim,net_dim],vf=[net_dim,net_dim])  # 策略网络和价值网络的层数和每层神经元数
        )
    # --- B. 搭建训练流水线 ---
    # 极其优雅且安全的单行写法：
    # 每次调用 lambda，都会向内存申请创建一个新的 Env，并立刻给它套上一个新的 Monitor
    env = DummyVecEnv([
        lambda: Monitor(gymnasium.make("lorenz_pmsm-v0"
                                    #    ,alpha=env_alpha
                                       ))
    ])

    env=VecNormalize(env,norm_obs=True,norm_reward=False,clip_obs=10.0)

    model=A2C(
        "MlpPolicy",
        env,
        learning_rate=lr,
        # ent_coef=ent_coef,
        n_steps=n_steps,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device= "cpu"
    )

    # --- C. 执行短平快的训练 (用于评估潜力) ---
    # 不需要跑满 50万步，跑 5万步足以看出这组参数有没有潜力
    try:
        model.learn(total_timesteps=50000)
    except Exception as e:
        # 如果参数太烂导致微分方程报错/梯度爆炸，直接返回一个极低的分数
        return -10000.0
    
    # --- D. 严格评估 ---
    # 测试时必须关闭环境内部的统计更新，否则评测结果会被训练时的统计数据污染
    eval_env=env
    eval_env.training=False
    eval_env.norm_reward=False

    # 用测试集跑 5 个回合，取平均奖励
    mean_reward,_=evaluate_policy(model,eval_env,n_eval_episodes=5, deterministic=True)

    return mean_reward

# ---------------------------------------------------------
# 3. 启动全局搜索
# ---------------------------------------------------------
if __name__ == "__main__":
    print("启动 Optuna 自动化调参引擎...")
    # 我们要“最大化”平均奖励
    study = optuna.create_study(direction="maximize")
    
    # 跑 50 组不同的参数组合（可以根据你的算力增加到 100 或 200）
    study.optimize(objective, n_trials=100)

    print("\n==================================")
    print("炼丹完成！最佳参数组合已找到：")
    print(f"历史最高平均奖励: {study.best_value}")
    
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("==================================")
