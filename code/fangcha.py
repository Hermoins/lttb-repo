import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 基础配置与模拟数据生成
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'PingFang SC', 'Microsoft YaHei', 'Arial Unicode MS', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
T = 3600
t = np.arange(T)

# 基础白噪声与呼吸效应
data = 0.3 + 0.05 * np.random.randn(T) + 0.1 * np.sin(2 * np.pi * t / 300)

# 植入三种典型异常模式 (带回落，更符合真实物理规律)
anomalies = np.zeros(T)
anomalies[1200:1210] = 0.8 * np.exp(-0.5 * np.arange(10))        # 模式A：突发尖峰
anomalies[1800:2100] = 0.005 * np.arange(300)                    # 模式B：缓慢涌出
anomalies[2100:2150] = 1.5 * np.exp(-0.1 * np.arange(50))        # 模式B 回落
anomalies[2700:2750] = 1.0 * (1 - np.exp(-0.05 * np.arange(50))) # 模式C：指数爆发
anomalies[2750:2800] = 1.0 * np.exp(-0.1 * np.arange(50))        # 模式C 回落
data = np.clip(data + anomalies, 0.1, 2.5)

# ==========================================
# 2. 计算滑动方差 (窗口=30)
# ==========================================
window_size = 30
V_array = np.zeros(T)
for i in range(T):
    start = max(0, i - window_size//2)
    end = min(T, i + window_size//2)
    V_array[i] = np.var(data[start:end])

# ==========================================
# 3. 传统固定桶 LTTB (死板对标，桶大小固定=50)
# ==========================================
idx_fix = [0]
pos = 0
B_fix = 50
while pos < T - 2:
    start = pos + 1
    end = min(start + B_fix, T - 1)
    if start >= end: break
    next_start = end
    next_end = min(next_start + B_fix, T)
    next_avg_x = (next_start + next_end) / 2
    next_avg_y = np.mean(data[next_start:next_end]) if next_start < next_end else data[-1]
    
    max_area = -1
    max_idx = start
    a_x, a_y = idx_fix[-1], data[idx_fix[-1]]
    for j in range(start, end):
        area = abs((a_x - next_avg_x) * (data[j] - a_y) - (a_x - j) * (next_avg_y - a_y)) * 0.5
        if area > max_area:
            max_area = area; max_idx = j
    idx_fix.append(max_idx)
    pos = end - 1

# ==========================================
# 4. 本发明：方差双阈值 + 流式秒级截断 LTTB
# ==========================================
idx_dyn = [0]
pos = 0
B_max = 80
B_min = 4
V_low = 0.005
V_high = 0.015

while pos < T - 2:
    start = pos + 1
    end = min(start + B_max, T - 1)
    
    # 【专利核心】：流式秒级截断！在等待桶装满的过程中，每一秒都在检查方差
    for j in range(start, end):
        v = V_array[j]
        if v > V_high:
            current_B = B_min
        elif v < V_low:
            current_B = B_max
        else:
            ratio = (v - V_low) / (V_high - V_low)
            current_B = int(np.round(B_max - ratio * (B_max - B_min)))
            
        # 如果当前桶的累积点数已经达到了此刻动态允许的最大尺寸，立刻拉闸截断！
        if (j - start + 1) >= current_B:
            end = j + 1
            break
            
    next_start = end
    next_end = min(next_start + B_max, T) 
    next_avg_x = (next_start + next_end) / 2
    next_avg_y = np.mean(data[next_start:next_end]) if next_start < next_end else data[-1]
    
    max_area = -1
    max_idx = start
    a_x, a_y = idx_dyn[-1], data[idx_dyn[-1]]
    for j in range(start, end):
        area = abs((a_x - next_avg_x) * (data[j] - a_y) - (a_x - j) * (next_avg_y - a_y)) * 0.5
        if area > max_area:
            max_area = area; max_idx = j
            
    idx_dyn.append(max_idx)
    pos = end - 1

# ==========================================
# 5. 掩码统计与打印量化结果
# ==========================================
anomaly_mask = np.zeros(T, dtype=bool)
# 定义三个异常区间
anomaly_mask[1200:1250] = True   
anomaly_mask[1800:2150] = True   
anomaly_mask[2700:2800] = True   
normal_mask = ~anomaly_mask

norm_fix = np.sum(normal_mask[idx_fix])
anom_fix = np.sum(anomaly_mask[idx_fix])
norm_dyn = np.sum(normal_mask[idx_dyn])
anom_dyn = np.sum(anomaly_mask[idx_dyn])

print("="*50)
print("📊 【点数大挪移】验证报告")
print("="*50)
print(f"📌 传统固定桶总点数: {len(idx_fix)}")
print(f"   - 平稳区(浪费): {norm_fix} 点")
print(f"   - 异常区(不足): {anom_fix} 点")
print("-" * 50)
print(f"📌 本发明动态桶总点数: {len(idx_dyn)}")
print(f"   - 平稳区(极其抠门): {norm_dyn} 点")
print(f"   - 异常区(火力全开): {anom_dyn} 点 (是传统的 {anom_dyn/anom_fix:.1f} 倍!)")
print("="*50)

# ==========================================
# 6. 可视化绘图
# ==========================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 1. 原始数据
axes[0].plot(t, data, 'k-', alpha=0.5, label='原始流式数据 (1Hz, 3600点)')
axes[0].set_title('图 1：原始矿井瓦斯浓度数据 (包含三种典型异常与高频噪声)', fontsize=13, fontweight='bold')
axes[0].legend(loc='upper left')
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0, 2.5)

# 2. 传统死板 LTTB
axes[1].plot(t, data, 'k-', alpha=0.2)
axes[1].plot(t[idx_fix], data[idx_fix], 'b.-', markersize=8, linewidth=1, 
             label=f'传统固定桶 (总点数:{len(idx_fix)} | 异常区点数:{anom_fix} | 平稳区点数:{norm_fix})')
axes[1].set_title('图 2：传统固定桶 LTTB (资源分配死板：在平稳区浪费带宽，在危险区采样严重不足)', fontsize=13, fontweight='bold')
axes[1].legend(loc='upper left')
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 2.5)

# 3. 本发明智能挪用 LTTB
axes[2].plot(t, data, 'k-', alpha=0.2)
axes[2].plot(t[idx_dyn], data[idx_dyn], 'r.-', markersize=8, linewidth=1, 
             label=f'本发明动态桶 (总点数:{len(idx_dyn)} | 异常区点数:{anom_dyn} | 平稳区点数:{norm_dyn})')
axes[2].axvspan(1200, 1250, color='orange', alpha=0.2, label='方差截断高频包抄区')
axes[2].axvspan(1800, 2150, color='orange', alpha=0.2)
axes[2].axvspan(2700, 2800, color='orange', alpha=0.2)
axes[2].set_title('图 3：本发明“方差双阈值”动态截断 LTTB (智能算力挪用：将平稳区的算力“挪用”到危险区高频包抄)', fontsize=13, fontweight='bold')
axes[2].legend(loc='upper left')
axes[2].grid(alpha=0.3)
axes[2].set_ylim(0, 2.5)
axes[2].set_xlabel('时间 (秒)', fontsize=12)

plt.tight_layout()
plt.savefig('variance_lttb_proof_v2.png', dpi=300, bbox_inches='tight')
plt.show()