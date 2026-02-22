import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def export_paper_figure(log_dir, target_metric, save_name):
    """
    从 TensorBoard 日志中提取数据，并绘制符合论文排版标准的高清图
    """
    print(f"正在读取日志文件夹: {log_dir} ...")
    
    # 1. 加载 TensorBoard 的黑匣子数据
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # 2. 检查你想要的指标存不存在
    available_metrics = ea.scalars.Keys()
    if target_metric not in available_metrics:
        print(f"❌ 找不到指标 '{target_metric}'。")
        print(f"当前可用的指标有: {available_metrics}")
        return
        
    # 3. 提取 X轴(步数) 和 Y轴(具体的数值)
    data = ea.scalars.Items(target_metric)
    steps = [event.step for event in data]
    values = [event.value for event in data]
    
    # ------------------ 开始绘制论文级别的图表 ------------------
    plt.figure(figsize=(8, 5)) # 设置图片大小比例
    
    # 画线：设置颜色为学术蓝，线条加粗
    plt.plot(steps, values, label="PPO with Attention", color="#1f77b4", linewidth=1.5)
    
    # 设置标题和坐标轴标签 (可以随意改成你要的英文或中文)
    plt.title("Lorenz System Control Performance", fontsize=14, fontweight='bold')
    plt.xlabel("Training Timesteps", fontsize=12)
    plt.ylabel("Average Reward (ep_rew_mean)", fontsize=12)
    
    # 添加网格线，让数据更容易看清
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 添加图例
    plt.legend(loc="lower right", fontsize=11)
    
    # 紧凑布局，防止坐标轴文字被裁切
    plt.tight_layout()
    
    # 4. 保存为超清 PDF 矢量图
    plt.savefig(save_name, format='pdf', dpi=300)
    plt.show()
    print(f"✅ 太棒了！高清图表已保存至: {os.path.abspath(save_name)}")

# ================= 使用方法 =================
if __name__ == "__main__":
    # 这里的路径要换成你真实跑出数据的那个 PPO_1 文件夹
    # 注意：一定要精确到包含 .tfevents 文件的那个最底层的文件夹！
    MY_LOG_FOLDER = "./lorenztensorboard2/PPO_1"  
    
    # PPO 最核心的奖励指标就是这个名字
    METRIC_TO_PLOT = "rollout/ep_rew_mean"
    
    # 输出的文件名
    OUTPUT_PDF = "Lorenz_Result.pdf"
    
    export_paper_figure(MY_LOG_FOLDER, METRIC_TO_PLOT, OUTPUT_PDF)