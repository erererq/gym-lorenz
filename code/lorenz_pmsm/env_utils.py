import sys
from pathlib import Path
current_dir=Path(__file__).resolve()

gym_lorenz_path=current_dir.parent.parent/"gym-lorenz"

sys.path.append(str(gym_lorenz_path))

import gym_lorenz
import gymnasium


def make_env(alpha=0.5,add_noise=False):
    """
    这个函数的作用是创建一个新的环境实例，并且可以通过参数来控制是否添加噪声，以及是否进入评估模式。
    
    参数说明：
    - add_noise: 布尔值，控制是否在环境中添加噪声。训练时可以随机添加不同强度的噪声，评估时可以锁定在最高难度。
    - eval_mode: 布尔值，控制环境是否进入评估模式。评估模式下，噪声强度固定，保证公平对待每一个模型。
    """
    env=gymnasium.make("lorenz_pmsm-v0",
                       alpha=alpha,
                       add_noise=add_noise
                       )
    return env