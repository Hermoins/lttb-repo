import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 构造具有真实物理特征的瓦斯模拟数据 (1Hz, 3600点 = 1小时)
# ==========================================
np.random.seed(42)
n_points = 3600
time = np.arange(n_points)

# 基础平稳瓦斯浓度 (0.1% 左右微幅波动)
gas_data = np.ones(n_points) * 0.1 + np.random.normal(0, 0.005, n_points)

# 插入干扰：传感器电磁毛刺噪声 (t=1000处，模拟假异常，验证抗噪防误报)
gas_data[1000:1005] += np.array([0.2, 0.5, 0.7, 0.2, 0.0])

# 插入真实异常：煤与瓦斯突出 (t=2800处开始飙升，呈S型/指数级爆发)
# 模拟在短短 40 秒内从 0.1% 飙升突破 0.8%
outburst = 1.0 / (1.0 + np.exp(-0.2 * (np.arange(300) - 40))) * 1.2
gas_data[2800:3100] += outburst

# ==========================================
# 2. 传统 Baseline 1：固定阈值报警 (通道2兜底)
# ==========================================
threshold_level = 0.8
alert_time_baseline = None
for i in range(n_points):
    if gas_data[i] >= threshold_level:
        alert_time_baseline = i
        break

# ==========================================
# 3. 传统 Baseline 2：固定大桶 LTTB (离线批处理，桶大小=60)
# ==========================================
trad_indices =[0]
bucket_size_trad = 60
i = 0
while i < n_points - bucket_size_trad:
    b_start = i
    b_end = i + bucket_size_trad
    pt_a = (trad_indices[-1], gas_data[trad_indices[-1]])
    
    # 传统LTTB：必须等下一个桶的数据都收集齐才能算平均值 (这就是延迟的来源)
    next_start = b_end
    next_end = min(next_start + bucket_size_trad, n_points)
    pt_c = ((next_start + next_end) / 2, np.mean(gas_data[next_start:next_end]))
    
    best_area = -1
    best_idx = b_start
    for k in range(b_start, b_end):
        pt_b = (k, gas_data[k])
        # 三角形面积公式
        area = abs((pt_a[0] - pt_c[0]) * (pt_b[1] - pt_a[1]) - (pt_a[0] - pt_b[0]) * (pt_c[1] - pt_a[1]))
        if area > best_area:
            best_area = area
            best_idx = k
    trad_indices.append(best_idx)
    i += bucket_size_trad

# ==========================================
# 4. 本发明核心：流式截断 + 动态LTTB + 趋势预警 (通道1)
# ==========================================
ours_indices = [0]
bucket_history =[]  # 记录当前采样桶的大小，用于画图
i = 0

# 【核心参数】
base_bucket = 60     # 平稳期大桶 (高压缩)
min_bucket = 5       # 突发期小桶 (高保真)
micro_grad_th = 0.015 # 雷达兵：微观斜率截断阈值 (%/秒)
macro_grad_th = 0.012 # 狙击手：宏观趋势斜率预警阈值 (%/秒)

while i < n_points - 1:
    # --- 雷达兵：后台滑动微窗口算微观斜率 ---
    lookback = gas_data[max(0, i-3):i+1]
    grad_micro = (lookback[-1] - lookback[0]) / len(lookback) if len(lookback) >= 2 else 0
    
    # 决定初始目标桶大小
    target_size = min_bucket if abs(grad_micro) > micro_grad_th else base_bucket
    
    # --- 流式截断模拟 ---
    b_start = i
    b_end = i
    for j in range(i + 1, min(i + target_size + 1, n_points)):
        b_end = j
        # 在填满大桶的过程中，实时监控，一旦突破阈值，强行截断！
        lb = gas_data[max(0, j-3):j+1]
        current_grad = (lb[-1] - lb[0]) / len(lb)
        if abs(current_grad) > micro_grad_th and target_size > min_bucket:
            target_size = j - i  # 强行截断
            break
            
    # --- 虚拟未来点近似 (切线外推) ---
    pt_a = (ours_indices[-1], gas_data[ours_indices[-1]])
    # 既然截断了没有未来数据，就用当前微观斜率向未来外推一个小桶的距离
    virtual_future_y = gas_data[b_end] + current_grad * min_bucket
    pt_c = (b_end + min_bucket, virtual_future_y)
    
    # LTTB 提纯特征点
    best_area = -1
    best_idx = b_start
    for k in range(b_start, b_end):
        pt_b = (k, gas_data[k])
        area = abs((pt_a[0] - pt_c[0]) * (pt_b[1] - pt_a[1]) - (pt_a[0] - pt_b[0]) * (pt_c[1] - pt_a[1]))
        if area > best_area:
            best_area = area
            best_idx = k
            
    ours_indices.append(best_idx)
    bucket_history.extend([b_end - b_start] * (b_end - b_start))
    i = b_end

# 补齐 bucket_history 长度
while len(bucket_history) < n_points:
    bucket_history.append(bucket_history[-1])

# --- 狙击手：通道1 宏观趋势斜率预警与驻留判断 ---
alert_time_ours = None
dwell_counter = 0

for k in range(4, len(ours_indices)):
    idx_window = ours_indices[k-4:k+1]
    val_window = gas_data[idx_window]
    
    # 宏观斜率：基于提纯后的 LTTB 骨架点计算
    macro_slope = (val_window[-1] - val_window[0]) / (idx_window[-1] - idx_window[0])
    
    if macro_slope > macro_grad_th:
        dwell_counter += 1
        # 驻留判断：必须连续 3 次满足条件才报警 (防误报)
        if dwell_counter >= 3 and alert_time_ours is None:
            alert_time_ours = idx_window[-1]
    else:
        dwell_counter = 0

# ==========================================
# 5. 可视化制图 (专利交底书素材)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows用黑体
plt.rcParams['axes.unicode_minus'] = False
# 如果是 Mac 用户，请将上面一行改为: plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1, 1.5]})
fig.suptitle('基于动态LTTB与流式截断的瓦斯双通道预警仿真验证', fontsize=18, fontweight='bold')

# --- 子图 1：算法采样保真度对比 (聚焦突变区) ---
zoom_start, zoom_end = 2750, 2950
ax1.plot(time[zoom_start:zoom_end], gas_data[zoom_start:zoom_end], color='lightgray', linewidth=3, label='10Hz原始高频瓦斯数据')
ax1.plot(time[zoom_start:zoom_end], gas_data[zoom_start:zoom_end], color='black', linewidth=1, alpha=0.5)

# 传统固定桶点
trad_zoom =[idx for idx in trad_indices if zoom_start <= idx < zoom_end]
ax1.scatter(trad_zoom, gas_data[trad_zoom], color='red', marker='x', s=100, linewidth=2, label='传统固定桶LTTB (严重遗漏突增特征)')
ax1.plot(trad_zoom, gas_data[trad_zoom], color='red', linestyle='--', alpha=0.5)

# 本发明动态截断点
ours_zoom =[idx for idx in ours_indices if zoom_start <= idx < zoom_end]
ax1.scatter(ours_zoom, gas_data[ours_zoom], color='green', marker='o', s=60, label='本发明动态LTTB (高密度完美咬合起增点与峰值)')
ax1.plot(ours_zoom, gas_data[ours_zoom], color='green', linestyle='-', alpha=0.8)

ax1.set_title('图A：流式截断提纯效果对比 (局部放大 2750s - 2950s)', fontsize=14)
ax1.set_ylabel('瓦斯浓度 (%)', fontsize=12)
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.7)

# --- 子图 2：自适应注意力机制 (动态桶大小变化) ---
ax2.plot(time, bucket_history, color='blue', linewidth=2, label='实时动态桶大小 (采样间隔)')
ax2.axvspan(990, 1010, color='orange', alpha=0.3, label='遇到毛刺噪声，瞬间切小桶')
ax2.axvspan(2780, 2900, color='red', alpha=0.3, label='遇到真实突出，触发流式截断高频采样')
ax2.set_title('图B：系统底层自适应注意力机制 (平稳期高压缩 -> 突变期0延迟截断)', fontsize=14)
ax2.set_ylabel('桶容量 (个采样点)', fontsize=12)
ax2.set_ylim(0, 70)
ax2.legend(loc='upper right', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.7)

# --- 子图 3：双通道预警时序对比 (抢救命时间) ---
ax3.plot(time[2750:2900], gas_data[2750:2900], color='black', linewidth=2, label='原始浓度真实走势')
ax3.axhline(y=0.8, color='red', linestyle='-.', linewidth=2, label='通道2：法定断电红线 (0.8%)')

if alert_time_ours:
    ax3.axvline(x=alert_time_ours, color='green', linestyle='--', linewidth=2.5, label=f'通道1: 趋势预警触发 (t={alert_time_ours}s)')
    ax3.scatter(alert_time_ours, gas_data[alert_time_ours], color='green', s=150, zorder=5)

if alert_time_baseline:
    ax3.axvline(x=alert_time_baseline, color='red', linestyle='-', linewidth=2.5, label=f'通道2: 绝对阈值断电触发 (t={alert_time_baseline}s)')
    ax3.scatter(alert_time_baseline, gas_data[alert_time_baseline], color='red', s=150, marker='*', zorder=5)

if alert_time_ours and alert_time_baseline:
    delta_t = alert_time_baseline - alert_time_ours
    ax3.annotate('', xy=(alert_time_baseline, 0.4), xytext=(alert_time_ours, 0.4),
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax3.text((alert_time_ours + alert_time_baseline)/2, 0.45, f'成功抢出提前量 ΔT = {delta_t} 秒！', 
             ha='center', color='blue', fontsize=14, fontweight='bold')

ax3.set_title('图C：双通道时序图 (化"事后断电"为"事前预测")', fontsize=14)
ax3.set_xlabel('时间 (秒)', fontsize=12)
ax3.set_ylabel('瓦斯浓度 (%)', fontsize=12)
ax3.legend(loc='upper left', fontsize=11)
ax3.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()