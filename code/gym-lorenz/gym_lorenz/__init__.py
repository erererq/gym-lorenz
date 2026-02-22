import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='lorenz_try-v0',
    entry_point="gym_lorenz.envs:lorenz_env_try",
    # 包路径:类名
    # entry_point='gym_lorenz.envs:lorenzEnv_transient',
    max_episode_steps = 2000,
    reward_threshold  = 1e50
    # kwargs={'alpha': 10.0, 'beta': 28.0}  # 可选参数
)
