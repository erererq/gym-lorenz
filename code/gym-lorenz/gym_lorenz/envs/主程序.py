import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') # 使用 Qt5 作为弹窗后端
import matplotlib.pyplot as plt

# --- 解决中文显示问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 用黑体
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac 用户打开这行
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 1. 设置 Figure (整个大画板) 的底色和边色
# ==========================================
# facecolor='lightblue' : 整个大画板铺满浅蓝色
# edgecolor='red'       : 整个大画板的边缘涂成红色
# linewidth=8           : 边框线加粗，不然太细看不清
fig, ax = plt.subplots(figsize=(8, 5), 
                       facecolor='lightblue', 
                       edgecolor='red', 
                       linewidth=8)

# ==========================================
# 2. 设置 Axes (画框/坐标系) 的底色
# ==========================================
# 把真正画数据的那个框框，底色填成浅黄色
ax.set_facecolor('lightyellow')

# ==========================================
# 3. 画柱状图，演示图形的底色和边色
# ==========================================
x = ['苹果', '香蕉', '橘子']
y = [3, 5, 2]

# facecolor (在 bar 里通常直接写 color) = 'lightgreen' -> 柱体内部是浅绿色
# edgecolor = 'black' -> 柱子的描边是黑色
# linewidth = 3 -> 描边加粗
ax.bar(x, y, 
       color='lightgreen',     # 柱子底色
       edgecolor='black',      # 柱子边色
       linewidth=3,            # 边色线条粗细
       linestyle='--')         # 边色线条改成虚线

# 加个标题
ax.set_title("秒懂【底色(Facecolor)】与【边色(Edgecolor)】", fontsize=16)

plt.show()