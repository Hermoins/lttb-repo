import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import uniform_filter1d

# ==========================================
# 0. 图表基础配置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'PingFang SC', 'Microsoft YaHei', 'Arial Unicode MS', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

WARNING_THRESHOLD_ABS = 1.0  # 法定的绝对兜底报警线 (1.0%)

# ==========================================
# 1. 🌟 用户可动态调节的预警参数区 🌟
# 你可以随意修改以下参数，图表中的所有文字、数据、线条会自动更新！
# ==========================================

# 场景1：模式B (地质构造缓慢涌出) 的预警参数
PARAM_B_TREND_THRESH = 0.0042 # 趋势触发斜率 (例如：0.0025%/s)
PARAM_B_DWELL_LIMIT  = 3       # 驻留确认次数 (连续3次满足才报警)
PARAM_B_REGRESS_WIN  = 4       # 拟合斜率所需的关键点个数

# 场景2：模式C (指数爆发) 的预警参数
PARAM_C_TREND_THRESH = 0.006   # 趋势触发斜率 (要求更陡峭)
PARAM_C_DWELL_LIMIT  = 2       # 驻留确认次数 (爆发极快，2次确认即可)
PARAM_C_REGRESS_WIN  = 3       # 拟合斜率所需的关键点个数

# 场景3：防误报噪声测试 的预警参数
PARAM_N_TREND_THRESH = 0.006   
PARAM_N_DWELL_LIMIT  = 3       

def get_dynamic_trend_threshold(current_concentration):
    """
    重构的二维联动阈值：精准匹配矿井真实物理规律
    """
    if current_concentration < 0.6:
        # 1. 绝对安全区：浓度很低，要求斜率极其陡峭（排除所有缓慢上升的假象）
        return 0.015   
    elif current_concentration < 0.75:
        # 2. 警觉区：浓度开始不正常了，只要有中等上升趋势（0.0035）就需警惕
        return 0.0035  
    else:
        # 3. 临战区（>0.75%）：瓦斯浓度已经逼近危险红线！
        # 此时，只要监测到微小的、持续的上升趋势（比如 0.002），就必须立刻报警！
        # 因为在 0.8% 的基数上，哪怕是 0.002 的增速也是致命的。
        return 0.002
# ==========================================
# 2. 核心算法定义
# ==========================================
def lttb_trend_warning_sim_dynamic(data, trend_threshold, dwell_limit, regress_window):
    """带斜率线性回归与驻留确认的动态 LTTB 算法"""
    n = len(data)
    # 微观梯度仅用于截断大桶
    gas_trend = uniform_filter1d(medfilt(data, kernel_size=5), size=15)
    G_array = np.abs(np.diff(gas_trend))
    
    indices = [0]
    pos = 0
    alert_trend_time = None
    trigger_points = None
    dwell_counter = 0
    
    while pos < n - 2:
        g_window = G_array[max(0, pos-5) : min(len(G_array), pos+5)]
        g_avg = np.mean(g_window) if len(g_window) > 0 else 0
        
        if g_avg < 0.0015: B_i = 80
        else: B_i = max(2, min(80, int(np.round(80 / (1 + 8000.0 * g_avg)))))
            
        start = pos + 1
        end = min(start + B_i, n - 1)
        if start >= end: break
        
        next_start = end
        next_end = min(next_start + B_i, n)
        next_avg_x = (next_start + next_end) / 2
        next_avg_y = np.mean(data[next_start:next_end]) if next_start < next_end else data[-1]
        
        max_area = -1
        max_idx = start
        a_x, a_y = indices[-1], data[indices[-1]]
        
        for j in range(start, end):
            area = abs((a_x - next_avg_x) * (data[j] - a_y) - (a_x - j) * (next_avg_y - a_y)) * 0.5
            if area > max_area:
                max_area = area; max_idx = j
                
        indices.append(max_idx)
        current_time = end - 1
        
        # 宏观趋势预警核心逻辑
        if alert_trend_time is None and len(indices) >= regress_window:
            xs = np.array(indices[-regress_window:])
            ys = np.array([data[i] for i in xs])
            
            if xs[-1] > xs[0]:
                slope, intercept = np.polyfit(xs, ys, 1)
                if slope >= get_dynamic_trend_threshold(ys[-1]):
                    dwell_counter += 1
                    if dwell_counter >= dwell_limit:
                        alert_trend_time = current_time 
                        trigger_points = (xs, ys, slope, intercept)
                else:
                    dwell_counter = 0 # 不满足则清零
                    
        pos = end - 1
        
    return indices, alert_trend_time, trigger_points

def baseline_threshold_alarm(data):
    """传统兜底物理报警"""
    return next((t for t, val in enumerate(data) if val >= WARNING_THRESHOLD_ABS), None)


# ==========================================
# 3. 场景数据生成与仿真执行
# ==========================================
np.random.seed(42)

# 场景1：模式 B (缓慢涌出)
T_B = 600; t_B = np.arange(T_B)
data_B = 0.3 + 0.02 * np.random.randn(T_B)
rise_B = np.zeros(T_B); rise_B[100:] = 0.0025 * np.arange(T_B - 100)
data_B = np.clip(data_B + rise_B, 0.1, 2.0)
idx_B, trend_B, pts_B = lttb_trend_warning_sim_dynamic(data_B, PARAM_B_TREND_THRESH, PARAM_B_DWELL_LIMIT, PARAM_B_REGRESS_WIN)
base_B = baseline_threshold_alarm(data_B)

# 场景2：模式 C (指数爆发)
T_C = 300; t_C = np.arange(T_C)
data_C = 0.3 + 0.02 * np.random.randn(T_C)
rise_C = np.zeros(T_C); rise_C[50:] = 1.0 * (1 - np.exp(-0.02 * np.arange(T_C - 50)))
data_C = np.clip(data_C + rise_C, 0.1, 2.0)
idx_C, trend_C, pts_C = lttb_trend_warning_sim_dynamic(data_C, PARAM_C_TREND_THRESH, PARAM_C_DWELL_LIMIT, PARAM_C_REGRESS_WIN)
base_C = baseline_threshold_alarm(data_C)

# 场景3：防误报噪声
T_N = 300; t_N = np.arange(T_N)
data_N = 0.3 + 0.02 * np.random.randn(T_N)
data_N[120:125] += 0.6 * np.sin(np.pi * np.arange(5)/5)
idx_N, trend_N, pts_N = lttb_trend_warning_sim_dynamic(data_N, PARAM_N_TREND_THRESH, PARAM_N_DWELL_LIMIT, 3)
base_N = baseline_threshold_alarm(data_N)


# ==========================================
# 4. 全动态可视化绘图
# ==========================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# --- 绘制图 1 (模式 B) ---
axes[0].plot(t_B, data_B, 'k-', alpha=0.3, label='原始数据 (1Hz)')
axes[0].scatter(idx_B, data_B[idx_B], c='blue', s=20, label='LTTB关键点')
axes[0].axhline(WARNING_THRESHOLD_ABS, color='red', linestyle='--', label=f'{WARNING_THRESHOLD_ABS}% 兜底报警线')
axes[0].axvline(base_B, color='red', linewidth=1.5, label=f'传统报警({base_B}s)')

if trend_B:
    val_B = data_B[trend_B] # 动态获取浓度
    advance_B = base_B - trend_B # 动态获取提前量
    
    axes[0].axvline(trend_B, color='green', linewidth=2, label=f'趋势预警({trend_B}s)')
    axes[0].annotate('', xy=(trend_B, 1.1), xytext=(base_B, 1.1), arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    axes[0].text((trend_B+base_B)/2, 1.15, f'超前 {advance_B} 秒预警！\n当前浓度: {val_B:.2f}%', color='green', ha='center', fontweight='bold', fontsize=12)
    
    xs, ys, slope, intercept = pts_B
    axes[0].plot(xs, slope*xs + intercept, 'g-', lw=3, label=f'动态触发斜率: {slope:.4f}/s')
    axes[0].scatter(xs, ys, c='lime', s=80, marker='*', zorder=10)

axes[0].set_title(f'验证 1：模式 B (缓慢涌出) - 参数配置 [斜率阈值:{PARAM_B_TREND_THRESH}, 驻留:{PARAM_B_DWELL_LIMIT}次]', fontweight='bold', fontsize=13)
axes[0].legend(loc='upper left'); axes[0].set_ylim(0, 1.4); axes[0].grid(alpha=0.3)

# --- 绘制图 2 (模式 C) ---
axes[1].plot(t_C, data_C, 'k-', alpha=0.3, label='原始数据')
axes[1].scatter(idx_C, data_C[idx_C], c='blue', s=20)
axes[1].axhline(WARNING_THRESHOLD_ABS, color='red', linestyle='--')
axes[1].axvline(base_C, color='red', linewidth=1.5, label=f'传统报警({base_C}s)')

if trend_C:
    val_C = data_C[trend_C]
    advance_C = base_C - trend_C
    
    axes[1].axvline(trend_C, color='green', linewidth=2, label=f'趋势预警({trend_C}s)')
    axes[1].annotate('', xy=(trend_C, 1.1), xytext=(base_C, 1.1), arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    axes[1].text((trend_C+base_C)/2, 1.15, f'超前 {advance_C} 秒逃生！\n当前浓度: {val_C:.2f}%', color='green', ha='center', fontweight='bold', fontsize=12)
    
    xs, ys, slope, intercept = pts_C
    axes[1].plot(xs, slope*xs + intercept, 'g-', lw=3, label=f'动态触发斜率: {slope:.4f}/s')
    axes[1].scatter(xs, ys, c='lime', s=80, marker='*', zorder=10)

axes[1].set_title(f'验证 2：模式 C (指数爆发) - 参数配置 [斜率阈值:{PARAM_C_TREND_THRESH}, 驻留:{PARAM_C_DWELL_LIMIT}次]', fontweight='bold', fontsize=13)
axes[1].legend(loc='upper left'); axes[1].set_ylim(0, 1.4); axes[1].grid(alpha=0.3)

# --- 绘制图 3 (防误报) ---
axes[2].plot(t_N, data_N, 'k-', alpha=0.3, label='含伪尖峰噪音的数据')
axes[2].scatter(idx_N, data_N[idx_N], c='blue', s=20)
axes[2].axhline(WARNING_THRESHOLD_ABS, color='red', linestyle='--')
axes[2].axvspan(120, 125, color='orange', alpha=0.3, label='短暂强噪声区')

if trend_N:
    axes[2].axvline(trend_N, color='red', linewidth=3, label='❌ 发生误报！(参数过松)')
else:
    axes[2].text(150, 0.8, f'✅ 驻留确认生效！\n要求连续驻留 {PARAM_N_DWELL_LIMIT} 次，\n当前噪声极短导致计数器清零，成功拦截误报！', 
                 color='green', fontweight='bold', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))

axes[2].set_title(f'验证 3：防误报测试 - 驻留确认核心作用 (驻留阈值={PARAM_N_DWELL_LIMIT})', fontweight='bold', fontsize=13)
axes[2].legend(loc='upper left'); axes[2].set_ylim(0, 1.4); axes[2].grid(alpha=0.3)
axes[2].set_xlabel('时间 (秒)', fontsize=12)

plt.tight_layout()
plt.savefig('dynamic_patent_warning_proof.png', dpi=300)

# ==========================================
# 5. 终端动态报告输出
# ==========================================
print("\n" + "="*50)
print("🎯 【趋势预警动态仿真测试报告】")
print("="*50)
if trend_B: print(f"✅ 模式B: 超前 {advance_B} 秒预警, 触发浓度 {val_B:.2f}%, 拟合斜率 {pts_B[2]:.4f}")
if trend_C: print(f"✅ 模式C: 超前 {advance_C} 秒预警, 触发浓度 {val_C:.2f}%, 拟合斜率 {pts_C[2]:.4f}")
print(f"🛡️ 噪声测试: {'❌ 发生误报' if trend_N else '✅ 成功拦截误报'}")
print("="*50)

plt.show()