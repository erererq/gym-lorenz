This is an environment for the transiently chaotic Lorenz system of equations, written in a OpenAI Gym compatible format.


Following should be installed,

1. TensorFlow
2. OpenAI Gym
3. Stable-baselines
4. numpy, matplotlib, scipy

As a first step, register the environment in the Gym (run "pip install -e ." in the gym-lorene folder). Follow the steps at the links below to get started,
a) https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa
b) Stable baselines documentation - https://stable-baselines.readthedocs.io/en/master/

Once, this has been done, run the file gym_run.py to begin training.

# 🧠 Synchronization of Hindmarsh-Rose Neural Models via PPO

本项目基于深度强化学习（PPO 算法）实现了 Hindmarsh-Rose (H-R) 混沌神经元模型的同步控制。代码不仅完成了对原论文的高精度复现，还在状态空间设计、特征提取和奖励重塑上进行了工业级优化。

## ✨ 核心优化与特性 (Features)

1. **自定义 Gym 环境封装 (`gym-lorenz`)**
   - 采用了标准的 Gymnasium 第三方包注册机制，实现了环境逻辑与训练/测试代码的完全解耦。
   - 底层物理引擎严格采用 **RK4 (四阶龙格-库塔法)** 进行高精度数值积分，有效镇压了混沌系统在欧拉法下的数值爆炸问题。
2. **6维“上帝视角”观测空间 (6D Observation Space)**
   - 突破了仅使用 3 维误差项导致的马尔可夫性缺失问题。
   - 将 Master 系统的 3 维绝对位置坐标与 3 维同步误差拼接，输入范围统一归一化至 `[-1, 1]`，极大提升了模型对非线性突变项的预测能力。
3. **自注意力机制特征提取 (Multihead Attention Extractor)**
   - 摒弃了传统的全连接层 (MLP) 提取器。
   - 将 6 维状态映射并切分为序列，利用 PyTorch 的 `MultiheadAttention` 动态捕捉“绝对位置”与“误差演变”之间的深层物理耦合关系。
4. **工业级全归一化奖励重塑 (Reward Shaping)**
   - 弃用物理量级差异巨大的原始奖励，采用全归一化空间计算法则。
   - 通过极佳的惩罚系数（如 `0.05`），在消除稳态误差与抑制控制力高频振颤（Chattering）之间找到了完美的帕累托前沿。

## 📂 目录结构 (Directory Structure)

```text
├── gym-lorenz/                  # 自定义环境库
│   ├── setup.py                 # 包安装配置文件
│   └── gym_lorenz/
│       ├── __init__.py          # 环境注册入口 (lorenz_try-v0)
│       └── envs/
│           ├── __init__.py      
│           └── lorenz_env_try.py # H-R 环境物理引擎与 Reward 逻辑
├── train.py                     # PPO 模型训练脚本
└── test_evaluate.py             # 测试、评估与论文 5 联图绘制脚本