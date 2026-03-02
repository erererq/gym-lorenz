import gymnasium 
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


import sys
from pathlib import Path
current_dir=Path(__file__).resolve()

gym_lorenz_path=current_dir.parent.parent/"gym-lorenz"

sys.path.append(str(gym_lorenz_path))

import gym_lorenz

# 让你的所有图表瞬间具备学术感
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

def calculate_advanced_metrics(arr_e, arr_a1, arr_a2, dt=0.01, error_band=0.05):
    """
    计算一系列高级指标来评估控制性能。
    
    参数说明：
    - arr_e: 误差数组 (shape: [steps, 3])，包含每个时间步的 e_d, e_q, e_w
    - arr_a1: 控制力 a1 数组 (shape: [steps,])，对应 d 轴的控制输入
    - arr_a2: 控制力 a2 数组 (shape: [steps,])，对应 q 轴的控制输入
    - dt: 时间步长，默认为 0.01 秒
    - error_band: 稳定带宽，默认为 0.05
    
    返回值：
    一个字典，包含以下指标：
    - "settling_time": 稳定时间，即误差进入并保持在稳定带内的时间
    - "overshoot": 超调量，即误差超过稳定带上限的最大值占稳定带上限的百分比
    - "control_effort": 控制努力，即所有时间步的 |a1| + |a2| 的总和乘以 dt
    """
    # 找出所有绝对误差大于允许频带的索引
    exceed_indices=np.where(np.abs(arr_e)>error_band)[0]
    if len(exceed_indices)==0:
        # 如果一上来误差就极小，且再也没出去过
        settling_time=0.0
    else:
        # 最后一个越界点的下一个索引，就是系统“彻底”稳定下来的时刻
        stable_idx = exceed_indices[-1] + 1
        # 如果直到测试结束，误差依然在外面震荡，说明未收敛
        settling_time = stable_idx * dt if stable_idx < len(arr_e) else np.nan
    # ==========================================
    # 2. 计算控制能量消耗 (Control Energy Cost)
    # 物理逻辑：计算整个控制过程中，控制力的平方的时间积分
    # 积分公式的离散化表达： sum(a1^2 + a2^2) * dt
    # ==========================================
    # 注意：这里的 arr_a1 和 arr_a2 必须是乘过放大系数(如 50 或 20)后的真实物理力量，而不是神经网络输出的 [-1, 1] 范围内的原始动作
    energy_cost = np.sum(np.square(arr_a1) + np.square(arr_a2)) * dt
    return settling_time, energy_cost
def test_evaluate_and_plot(model_path, vecnorm_path, num_tests=10, steps=2000, dt=0.001,alpha=0.5):
    # 1. 挂载环境与归一化包装器
    env = DummyVecEnv([
        lambda: Monitor(gymnasium.make("lorenz_pmsm-v0",alpha=alpha))
    ])

    env=VecNormalize.load(vecnorm_path,env)

    # 【核心】：关闭更新
    env.training = False      
    env.norm_reward = False   
    
    # 2. 加载模型
    model = A2C.load(model_path, env=env)
    
    # 3. 数据大容器
    all_e1, all_e2, all_e3 = [], [], []
    all_u1, all_u2 = [], []
    initial_states = [] # 记录每次测试的初始误差，用于图例
    
    mae_list, rmse_list, ts_list, energy_list = [], [], [], []

    print(f"开始进行 {num_tests} 组随机初始条件的测试...")
    # 4. 跑批测试循环
    for j in range(num_tests):
        obs = env.reset()
        base_env = env.envs[0].unwrapped # 穿透 DummyVecEnv 和 Monitor 获取最底层物理环境
        
        # 记录初始物理误差
        init_e1 = base_env.state1[0] - base_env.state2[0]
        init_e2 = base_env.state1[1] - base_env.state2[1]
        init_e3 = base_env.state1[2] - base_env.state2[2]
        initial_states.append([init_e1, init_e2, init_e3])
        
        list_e1, list_e2, list_e3 = [], [], []
        list_u1, list_u2 = [], []

        for i in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # 抓取真实物理误差
            real_e1 = base_env.state1[0] - base_env.state2[0]
            real_e2 = base_env.state1[1] - base_env.state2[1]
            real_e3 = base_env.state1[2] - base_env.state2[2]
            
            # 抓取真实控制力 (假设网络输出 [-1,1]，物理映射为 f_max=50)
            real_u1 = action[0][0] * 50.0 
            real_u2 = action[0][1] * 50.0 
            
            list_e1.append(real_e1)
            list_e2.append(real_e2)
            list_e3.append(real_e3)
            list_u1.append(real_u1)
            list_u2.append(real_u2)

        # 转换为 numpy 数组
        arr_e1, arr_e2, arr_e3 = np.array(list_e1), np.array(list_e2), np.array(list_e3)
        arr_u1, arr_u2 = np.array(list_u1), np.array(list_u2)

        
        # 保存轨迹用于画图
        all_e1.append(arr_e1)
        all_e2.append(arr_e2)
        all_e3.append(arr_e3)
        all_u1.append(arr_u1)
        all_u2.append(arr_u2)
        
        # 计算该回合指标
        # 假设 steps = 500 (5秒)
        # 安全的截取法：如果总步数小于要求，就取后一半作为稳态
        actual_steady_start = min(1000, len(arr_e1) // 2) 
        steady_e1 = arr_e1[actual_steady_start:]
        steady_e2 = arr_e2[actual_steady_start:]
        steady_e3 = arr_e3[actual_steady_start:]
        mae = np.mean([np.mean(np.abs(steady_e1)), np.mean(np.abs(steady_e2)), np.mean(np.abs(steady_e3))])
        rmse = np.mean([np.sqrt(np.mean(steady_e1**2)), np.sqrt(np.mean(steady_e2**2)), np.sqrt(np.mean(steady_e3**2))])
        
        # 计算 e1, e2, e3 中最晚收敛的时间作为系统整体收敛时间
        ts1, energy = calculate_advanced_metrics(arr_e1, arr_u1, arr_u2, dt=dt)
        ts2, _ = calculate_advanced_metrics(arr_e2, arr_u1, arr_u2, dt=dt)
        ts3, _ = calculate_advanced_metrics(arr_e3, arr_u1, arr_u2, dt=dt)
        max_ts = np.nanmax([ts1, ts2, ts3]) 
        
        mae_list.append(mae)
        rmse_list.append(rmse)
        ts_list.append(max_ts)
        energy_list.append(energy)

    # 5. 打印综合成绩单
    print("\n" + "="*40)
    print("🎯 模型综合性能评估报告 (10回合平均)")
    print("="*40)
    print(f"Avg MAE (平均绝对误差)  : {np.mean(mae_list):.4f}")
    print(f"Avg RMSE (均方根误差)   : {np.mean(rmse_list):.4f}")
    
    valid_ts = [t for t in ts_list if not np.isnan(t)]
    if valid_ts:
        print(f"Avg Settling Time (平均调节时间) : {np.mean(valid_ts):.3f} s")
        print(f"收敛成功率 : {len(valid_ts)}/{num_tests}")
    else:
        print("Avg Settling Time : 未能收敛！")
        
    print(f"Avg Control Energy (平均控制能耗) : {np.mean(energy_list):.2f}")
    print("="*40 + "\n")

    # 只需要传数据和想要显示的标签名称！
    plot_3d_surface(all_e1, z_label='$x_1$')
    plot_3d_surface(all_e2, z_label='$x_2$')
    plot_3d_surface(all_e3, z_label='$x_3$')
    

# def plot_3d_error_trajectory(arr_e1, arr_e2, arr_e3):
#     """
#     绘制论文同款的三维误差同步相图 (3D Phase Portrait)
#     """
#     # 开启一个 3D 画布
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # 1. 绘制误差轨迹线
#     # 用从浅到深的颜色渐变（或者单色）画出轨迹
#     ax.plot(arr_e1, arr_e2, arr_e3, color='blue', linewidth=1.2, label='Error Trajectory')
    
#     # 2. 标出起点 (Start) 和 终点 (End)
#     ax.scatter(arr_e1[0], arr_e2[0], arr_e3[0], 
#                color='red', s=80, marker='o', label='Start Point', zorder=5)
    
#     ax.scatter(arr_e1[-1], arr_e2[-1], arr_e3[-1], 
#                color='green', s=150, marker='*', label='End Point (Origin)', zorder=5)
    
#     # 3. 画出空间直角坐标系的虚线基准十字星 (相交于原点 0,0,0)
#     ax.plot([min(arr_e1), max(arr_e1)], [0, 0], [0, 0], color='gray', linestyle='--', linewidth=1)
#     ax.plot([0, 0], [min(arr_e2), max(arr_e2)], [0, 0], color='gray', linestyle='--', linewidth=1)
#     ax.plot([0, 0], [0, 0], [min(arr_e3), max(arr_e3)], color='gray', linestyle='--', linewidth=1)
    
#     # 4. 设置视角和标签
#     ax.set_xlabel('Error $e_1$ (d-axis)', fontsize=12, labelpad=10)
#     ax.set_ylabel('Error $e_2$ (q-axis)', fontsize=12, labelpad=10)
#     ax.set_zlabel('Error $e_3$ ($\omega$-axis)', fontsize=12, labelpad=10)
#     ax.set_title('3D Phase Portrait of Synchronization Errors', fontsize=15)
    
#     # 调整初始观察视角 (仰角 25 度，方位角 45 度)
#     ax.view_init(elev=25, azim=45)
    
#     ax.legend(fontsize=11)
#     plt.tight_layout()
#     plt.show()

def plot_3d_surface(e,z_label='$x_1$',dt=0.001, display_seconds=2.0):
    """
    绘制多条误差轨迹的三维相图
    """
    # 假设 Z 就是你的误差矩阵，形状为 (20, 200)
    Z_full=np.array(e)
    display_steps=int(display_seconds/dt)
    Z=Z_full[:,:display_steps] # 只显示前 display_seconds 秒的数据
    num_tests,num_steps=Z.shape
    # 1. 定义 X 轴 (时间) 和 Y 轴 (测试编号) 的一维坐标
    x=np.linspace(0,2,num_steps)
    y=np.arange(1, num_tests+1)
    # 2. 【核心魔法】生成 2D 坐标网格
    X,Y=np.meshgrid(x,y)
    # 现在 X 和 Y 都变成了 (20, 200) 的二维矩阵，和 Z 完美对应！
    fig = plt.figure(figsize=(10,8))
    # 声明这是一个 3D 坐标系
    ax=fig.add_subplot(111,projection='3d')
    # 调整视角 (仰角 15 度，方位角 -60 度，这是这种图最经典的展示视角)
    ax.view_init(elev=15,azim=-60)
    # cmap='jet' 是科研中最常用的彩虹色带（红高蓝低）
    # rstride 和 cstride 控制网格的疏密，设为 1 表示最细腻
    # alpha 控制透明度，稍微透明一点能透出底下的阴影
    surf = ax.plot_surface(X,Y,Z,cmap="jet",
                           rstride=1,cstride=1,
                           linewidth=0,antialiased=True,alpha=0.9)
    # offset 设置了影子投射的高度位置。我们把它投射到 Z 轴的最低点（地平线）
    z_min=np.min(Z)

    # zdir='z' 表示向 Z 轴方向投影
    cset = ax.contourf(X,Y,Z,zdir="z",offset=z_min,cmap="jet",alpha=0.7)

    # 把 Z 轴的下限锁死在影子的位置，防止图形悬空
    ax.set_zlim(z_min,np.max(Z))

    ax.set_xlabel('t/s', fontsize=14, labelpad=10)
    ax.set_ylabel('test', fontsize=14, labelpad=10)
    ax.set_zlabel('$x_1$', fontsize=14, labelpad=10) # 论文里误差用的是 x1, x2, x3

    # 添加颜色条，shrink 控制它的大小，pad 控制它离主图的距离
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)

    plt.show()

if __name__ == "__main__":
    # 调用此函数即可一键完成测试、出报告并弹出图表
    test_evaluate_and_plot(
        model_path="pmsm_attention_clean_model",     # 你的模型名称
        vecnorm_path="pmsm_attention_clean_vecnorm.pkl", # 你的归一化文件
        num_tests=10, 
        steps=2000,
        alpha=0.5
    )
    