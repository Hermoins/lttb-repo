import numpy as np
import matplotlib.pyplot as plt

# # 配置中文显示
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams["axes.unicode_minus"] = False

# # 第一阶段：平稳期（0-4000点）
# t1 = np.linspace(0, 4000, 4000)
# c1 = 0.3 + 0.1 * np.random.randn(4000)  # 均值0.3%，标准差0.1%

# # 第二阶段：缓慢上升期（4000-7000点）
# t2 = np.linspace(4000, 7000, 3000)
# c2 = 0.3 + 0.002 * (t2 - 4000) + 0.05 * np.random.randn(3000)

# # 第三阶段：异常突增期（7000-10000点）
# t3 = np.linspace(7000, 10000, 3000)
# c3 = 0.4 + 1.1 * (1 - np.exp(-0.002 * (t3 - 7000))) + 0.1 * np.random.randn(3000)

# # 添加脉冲噪声（模拟传感器干扰）
# pulse_indices = np.random.choice(10000, 50, replace=False)
# c_combined = np.concatenate([c1, c2, c3])
# c_combined[pulse_indices] += 0.5 * np.random.randn(50)  # 添加50个噪声点

# # 验证结果：打印关键信息
# print("生成的瓦斯数据总长度：", len(c_combined))
# print("前5个数据点（平稳期）：", c_combined[:5])
# print("4000-4005个数据点（上升期）：", c_combined[4000:4005])
# print("9995-10000个数据点（突增期）：", c_combined[9995:])


# # 绘制瓦斯浓度趋势图
# plt.figure(figsize=(12, 6))
# plt.plot(c_combined, color='blue', linewidth=0.8)
# plt.axvline(x=4000, color='orange', linestyle='--', label='平稳期→上升期')
# plt.axvline(x=7000, color='red', linestyle='--', label='上升期→突增期')
# plt.xlabel('时间（秒）')
# plt.ylabel('瓦斯浓度（%）')
# plt.title('合成瓦斯浓度变化趋势')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # 解决Matplotlib中文显示问题
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams["axes.unicode_minus"] = False

# # 生成10000个数据点，采样率1Hz（对应10000秒，约2.78小时）
# np.random.seed(42)  # 设置随机种子，保证结果可复现

# # 第一阶段：正常平稳期（0-6000点，约1.67小时）
# # 真实煤矿正常作业时瓦斯浓度：均值0.3%，小幅波动（标准差0.05%）
# t1 = np.linspace(0, 6000, 6000)
# c1 = 0.3 + 0.05 * np.random.randn(6000)
# # 限制浓度下限为0（瓦斯浓度不可能为负）
# c1 = np.clip(c1, 0, None)

# # 第二阶段：缓慢上升期（6000-8500点，约0.69小时）
# # 从0.3%缓慢上升至1.0%（预警值），波动更小（标准差0.03%）
# t2 = np.linspace(6000, 8500, 2500)
# c2 = 0.3 + 0.00028 * (t2 - 6000) + 0.03 * np.random.randn(2500)
# c2 = np.clip(c2, 0, None)

# # 第三阶段：异常上升期（8500-10000点，约0.42小时）
# # 从1.0%快速上升至1.8%（高风险，但未达爆炸极限），波动略增（标准差0.08%）
# t3 = np.linspace(8500, 10000, 1500)
# # 用更平缓的指数函数模拟瓦斯逸出的真实上升规律
# c3 = 1.0 + 0.8 * (1 - np.exp(-0.0015 * (t3 - 8500))) + 0.08 * np.random.randn(1500)
# c3 = np.clip(c3, 0, None)

# # 添加真实传感器噪声（脉冲干扰更少、幅度更小）
# # 真实传感器脉冲噪声通常只有10-20个，幅度0.1%左右
# pulse_indices = np.random.choice(10000, 15, replace=False)
# c_combined = np.concatenate([c1, c2, c3])
# c_combined[pulse_indices] += 0.1 * np.random.randn(15)
# # 再次限制浓度，避免脉冲噪声导致异常高值
# c_combined = np.clip(c_combined, 0, 2.0)

# # 绘制真实化的瓦斯浓度趋势图
# plt.figure(figsize=(12, 6))
# plt.plot(c_combined, color='#1f77b4', linewidth=0.8, label='瓦斯浓度')
# # 标注安全阈值线
# plt.axhline(y=1.0, color='#ff7f0e', linestyle='--', linewidth=1.2, label='预警值（1.0%）')
# plt.axhline(y=1.5, color='#d62728', linestyle='--', linewidth=1.2, label='危险值（1.5%）')
# # 标注阶段分割线
# plt.axvline(x=6000, color='gray', linestyle=':', label='平稳期→上升期')
# plt.axvline(x=8500, color='gray', linestyle=':', label='上升期→异常期')

# plt.xlabel('时间（秒）', fontsize=11)
# plt.ylabel('瓦斯浓度（%）', fontsize=11)
# plt.title('煤矿真实场景瓦斯浓度模拟数据（10000点，1Hz）', fontsize=13)
# plt.legend(loc='upper left')
# plt.grid(alpha=0.3)
# # 限定y轴范围，更贴合实际
# plt.ylim(0, 2.2)
# plt.show()

# # 输出关键统计信息，验证数据合理性
# print("=== 模拟数据关键统计 ===")
# print(f"数据总长度：{len(c_combined)} 点")
# print(f"平稳期浓度均值：{np.mean(c1):.3f}%，最大值：{np.max(c1):.3f}%")
# print(f"上升期浓度均值：{np.mean(c2):.3f}%，最大值：{np.max(c2):.3f}%")
# print(f"异常期浓度均值：{np.mean(c3):.3f}%，最大值：{np.max(c3):.3f}%")
# print(f"整体数据最大值：{np.max(c_combined):.3f}%（未达爆炸极限）")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 解决Matplotlib中文显示问题（关键：避免标签乱码）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 基础参数
Fs = 1  # 采样频率 1Hz
T = 3600  # 模拟1小时，3600个数据点
t = np.arange(0, T, 1/Fs)  # 生成0-3599秒的时间轴，共3600个点

# 1. 平稳基线 + 随机噪声（真实煤矿正常浓度）
base = 0.3 + 0.05 * np.random.randn(T)  # 均值0.3%，标准差0.05%

# 2. 添加周期性呼吸波动（采煤机周期性影响）
# 周期5分钟（300秒），幅度0.1%，模拟采煤机工作的周期性扰动
breath = 0.1 * np.sin(2*np.pi*t/300)  

# 3. 定义三种异常模式（贴合煤矿真实异常场景）
anomalies = np.zeros(T)

# 模式A：突发尖峰（放顶煤引起）@ t=1200s，10秒内快速衰减
anomalies[1200:1210] = 0.8 * np.exp(-0.5*(np.arange(10)))  

# 模式B：缓慢持续涌出（地质构造）@ t=1800-2100s，5分钟线性增长至1.5%
anomalies[1800:2100] = 0.005 * (np.arange(300))  

# 模式C：指数型爆发（揭穿瓦斯包）@ t=2700-2750s，50秒指数增长至≈1.2%
anomalies[2700:2750] = 1.2 * (1 - np.exp(-0.1*(np.arange(50))))  

# 4. 添加脉冲噪声（模拟传感器干扰，20个噪声点，幅度0.3%）
pulse_noise = np.zeros(T)
pulse_indices = np.random.choice(T, 20, replace=False)
pulse_noise[pulse_indices] = 0.3 * np.random.randn(20)

# 最终合成数据
gas_concentration = base + breath + anomalies + pulse_noise
# 限制浓度范围：0.1%（最低检测值）~2.0%（高风险但未达爆炸极限）
gas_concentration = np.clip(gas_concentration, 0.1, 2.0)  

# ========== 可视化结果（直观查看数据特征） ==========
plt.figure(figsize=(14, 7))
plt.plot(t, gas_concentration, color='#1f77b4', linewidth=0.8, label='瓦斯浓度')

# 标注三种异常模式
plt.axvspan(1200, 1210, alpha=0.2, color='red', label='模式A：放顶煤尖峰')
plt.axvspan(1800, 2100, alpha=0.2, color='orange', label='模式B：地质构造涌出')
plt.axvspan(2700, 2750, alpha=0.2, color='purple', label='模式C：揭穿瓦斯包')

# 标注安全阈值
plt.axhline(y=1.0, color='#ff7f0e', linestyle='--', linewidth=1.2, label='预警值（1.0%）')
plt.axhline(y=1.5, color='#d62728', linestyle='--', linewidth=1.2, label='危险值（1.5%）')

plt.xlabel('时间（秒）', fontsize=11)
plt.ylabel('瓦斯浓度（%）', fontsize=11)
plt.title('1小时煤矿瓦斯浓度模拟数据（含3种典型异常模式）', fontsize=13)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.xlim(0, 3600)
plt.ylim(0, 2.2)
plt.show()

# ========== 输出关键统计信息（验证数据合理性） ==========
print("=== 瓦斯模拟数据统计 ===")
print(f"数据总长度：{len(gas_concentration)} 点（1小时，1Hz采样）")
print(f"浓度均值：{np.mean(gas_concentration):.3f}%")
print(f"浓度最大值：{np.max(gas_concentration):.3f}%")
print(f"浓度最小值：{np.min(gas_concentration):.3f}%")

# 可选：将数据保存为CSV文件，方便后续分析/建模
df = pd.DataFrame({
    '时间(秒)': t,
    '瓦斯浓度(%)': gas_concentration
})
df.to_csv('煤矿瓦斯模拟数据_1小时.csv', index=False, encoding='utf-8')
print("\n数据已保存为：煤矿瓦斯模拟数据_1小时.csv")