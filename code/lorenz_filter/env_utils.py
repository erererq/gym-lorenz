import sys
from pathlib import Path
current_dir=Path(__file__).parent.resolve()

gym_lorenz_path=current_dir.parent.parent/"gym-lorenz"

sys.path.append(str(gym_lorenz_path))

import gym_lorenz
import gymnasium
from gymnasium.wrappers import FrameStackObservation

def make_env(add_noise=False,eval_mode=False,add_filter=False,use_frame_stack=True):
    """
    这个函数的作用是创建一个新的环境实例，并且可以通过参数来控制是否添加噪声、是否进入评估模式，以及是否启用动作滤波器。
    
    参数说明：
    - add_noise: 布尔值，控制是否在环境中添加噪声。训练时可以随机添加不同强度的噪声，评估时可以锁定在最高难度。
    - eval_mode: 布尔值，控制环境是否进入评估模式。评估模式下，噪声强度固定，保证公平对待每一个模型。
    - add_filter: 布尔值，控制是否启用动作滤波器。启用后，环境会对 RL Agent 输出的动作进行平滑处理，防止过于激烈的动作导致系统发散。
    """
    env=gymnasium.make("lorenz_try-v0",
                       add_noise=add_noise,
                       eval_mode=eval_mode,
                       add_filter=add_filter)
    return env