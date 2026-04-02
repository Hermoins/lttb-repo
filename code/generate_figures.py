"""
################################################################################
# 专利附图生成脚本
# ================================================================================
# 
# 本脚本用于生成专利申请所需的7张附图，展示了本发明的核心技术和效果
# 
# 生成图片列表：
#   1. fig1_flowchart.png   - 方法总体流程图
#   2. fig2_raw_data.png    - 模拟瓦斯数据（含三种异常模式）
#   3. fig3_bucket_adjustment.png - 动态桶调整策略示意图
#   4. fig4_comparison.png  - 传统LTTB与本发明效果对比
#   5. fig5_timing.png     - 预警时间对比示意图
#   6. fig6_architecture.png - 系统架构示意图
#   7. fig7_denoise.png    - 抗噪机制效果图
#
# 依赖库：numpy, matplotlib, scipy
# 运行方式：python generate_figures.py
#
################################################################################
"""

# 导入必要的库
import numpy as np          # 数值计算库
import matplotlib.pyplot as plt  # 绘图库
import matplotlib.patches as mpatches  # 图形补丁（用于绘制特殊形状）
from matplotlib.patches import FancyBboxPatch  # 圆角矩形框
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# 设置中文字体，确保matplotlib能正确显示中文
# Windows常用: SimHei, Microsoft YaHei
# Linux常用: WenQuanYi Micro Hei
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ================================================================================
# 图1：算法流程图
# ================================================================================
def generate_figure1_flowchart():
    """
    图1：方法总体流程图
    
    本图展示本发明的核心技术流程，包括：
    1. 数据采集 -> 2. 抗噪预处理 -> 3. 变化率计算 -> 4. 动态桶调整 -> 
    5. LTTB下采样 -> 6. 双模式预警 -> 7. 输出预警信号
    
    同时展示"流式截断"分支机制，说明在检测到异常时可立即切换采样策略
    """
    
    # 创建画布：14x8英寸，dpi 100(默认)
    # fig是图形对象，ax是坐标轴对象
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # 设置坐标轴范围：x轴0-14，y轴0-10
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    # 关闭坐标轴显示（因为是流程图，不需要坐标轴刻度）
    ax.axis('off')
    
    # ================================================================================
    # 定义绘图辅助函数：绘制圆角矩形框
    # ================================================================================
    def draw_box(x, y, w, h, text, color='#E8F4FD'):
        """
        绘制圆角矩形框（流程图节点）
        
        参数:
            x, y   : 框中心坐标
            w, h   : 框的宽度和高度
            text   : 框内显示的文字
            color  : 框的填充颜色
        """
        # 创建圆角矩形对象
        # boxstyle="round,pad=0.05" 表示圆角，文字与边框间距0.05
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, 
                             boxstyle="round,pad=0.05", 
                             facecolor=color,      # 填充颜色
                             edgecolor='#2C82C9', # 边框颜色
                             linewidth=2)         # 边框宽度
        ax.add_patch(box)  # 将框添加到坐标轴
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # ================================================================================
    # 定义绘图辅助函数：绘制箭头
    # ================================================================================
    def draw_arrow(x1, y1, x2, y2):
        """
        绘制带箭头的连线
        
        参数:
            x1, y1 : 箭头起点坐标
            x2, y2 : 箭头终点坐标
        """
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='#2C82C9', lw=2))
    
    # ================================================================================
    # 绘制主流程框（从下往上绘制）
    # ================================================================================
    
    # 步骤1：数据采集
    # 显示采样频率1Hz-10Hz
    draw_box(7, 9, 2.5, 0.8, '数据采集\n(1Hz-10Hz)', '#B8D4E8')
    draw_arrow(7, 8.6, 7, 8.2)  # 箭头指向下一步
    
    # 步骤2：抗噪预处理
    # 使用中值滤波去除传感器毛刺
    draw_box(7, 8, 2.5, 0.8, '抗噪预处理\n(中值滤波)', '#B8D4E8')
    draw_arrow(7, 7.6, 7, 7.2)
    
    # 步骤3：变化率计算
    # 计算梯度和方差作为变化率指标
    draw_box(7, 7, 2.5, 0.8, '变化率计算\n(梯度/方差)', '#B8D4E8')
    draw_arrow(7, 6.6, 7, 6.2)
    
    # 步骤4：动态桶调整（核心创新）- 用特殊颜色突出
    # 根据变化率实时调整LTTB的桶大小
    draw_box(7, 6, 2.5, 0.8, '动态桶调整\n(弹性采样)', '#FCD5B4')
    draw_arrow(7, 5.6, 7, 5.2)
    
    # 步骤5：LTTB下采样
    # 基于动态桶执行LTTB算法提取关键点
    draw_box(7, 5, 2.5, 0.8, 'LTTB下采样\n(关键点提取)', '#B8D4E8')
    draw_arrow(7, 4.6, 7, 4.2)
    
    # 步骤6：双模式预警 - 阈值报警 + 趋势预警
    draw_box(7, 4, 2.5, 0.8, '双模式预警\n(阈值+趋势)', '#C8E6C9')
    draw_arrow(7, 3.6, 7, 3.2)
    
    # 步骤7：最终输出
    draw_box(7, 3, 2.5, 0.8, '输出预警信号', '#C8E6C9')
    
    # ================================================================================
    # 绘制流式截断分支（右侧）
    # ================================================================================
    # 当检测到异常时，立即截断大桶切换为小桶
    draw_box(11, 6, 1.8, 0.6, '流式截断', '#FFE0B2')
    draw_arrow(8.75, 6, 10.1, 6)  # 从动态桶调整分支出来
    
    ax.text(12.4, 6, '检测异常\n立即切换', fontsize=9, ha='left', va='center')
    
    # 设置图表标题
    ax.set_title('图1：本发明方法总体流程图', fontsize=14, fontweight='bold', pad=20)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('fig1_flowchart.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("图1已生成: fig1_flowchart.png")


# ================================================================================
# 图2：原始数据与异常模式
# ================================================================================
def generate_figure2_raw_data():
    """
    图2：模拟瓦斯数据与三种异常模式
    
    本图生成1小时的模拟瓦斯数据，包含三种典型异常模式：
    - 模式A：突发尖峰（放顶煤引起），10秒内快速上升后衰减
    - 模式B：缓慢涌出（地质构造），5分钟内线性增长
    - 模式C：指数爆发（揭穿瓦斯包），50秒内指数增长
    
    同时添加了传感器噪声和周期性波动，模拟真实煤矿场景
    """
    
    # 设置随机种子，确保每次生成的数据一致（可重复验证）
    np.random.seed(42)
    
    # ================================================================================
    # 参数设置
    # ================================================================================
    Fs = 1        # 采样频率：1Hz（每秒1个数据点）
    T = 3600      # 总时长：3600秒（1小时）
    t = np.arange(0, T, 1/Fs)  # 时间轴：0, 1, 2, ..., 3599
    
    # ================================================================================
    # 生成基础数据（平稳期的正常波动）
    # ================================================================================
    # 均值0.3%，标准差0.05%的高斯噪声（正常瓦斯浓度范围）
    base = 0.3 + 0.05 * np.random.randn(T)
    
    # 周期性呼吸波动：模拟采煤机周期性工作对瓦斯浓度的影响
    # 周期300秒（5分钟），幅度0.1%
    breath = 0.1 * np.sin(2 * np.pi * t / 300)
    
    # ================================================================================
    # 添加三种异常模式
    # ================================================================================
    anomalies = np.zeros(T)  # 初始化异常数组（全零）
    
    # 模式A：突发尖峰 @ 1200-1210秒（放顶煤引起）
    # 10秒内从0.3%上升至1.1%（0.8%的增量），然后指数衰减
    # 使用指数函数 exp(-0.5 * t) 模拟快速衰减
    anomalies[1200:1210] = 0.8 * np.exp(-0.5 * np.arange(10))  # 模式A
    
    # 模式B：缓慢涌出 @ 1800-2100秒（地质构造引起）
    # 300秒（5分钟）内线性增长，从0.3%增长到1.8%
    # 斜率 = 0.005%/秒
    anomalies[1800:2100] = 0.005 * np.arange(300)  # 模式B
    
    # 模式C：指数爆发 @ 2700-2750秒（揭穿瓦斯包）
    # 50秒内指数增长至约1.5%，使用公式: A * (1 - exp(-k*t))
    anomalies[2700:2750] = 1.2 * (1 - np.exp(-0.1 * np.arange(50)))  # 模式C
    
    # ================================================================================
    # 添加传感器脉冲噪声（电磁干扰等）
    # ================================================================================
    pulse_noise = np.zeros(T)
    # 随机选择20个位置添加噪声
    pulse_noise[np.random.choice(T, 20, replace=False)] = 0.3 * np.random.randn(20)
    
    # ================================================================================
    # 合成最终数据并进行范围限制
    # ================================================================================
    # 合成：基础 + 呼吸波动 + 异常 + 噪声
    gas_data = base + breath + anomalies + pulse_noise
    # 限制浓度范围：0.1%（最低检测值）~ 2.0%（高风险值，未达爆炸极限）
    gas_data = np.clip(gas_data, 0.1, 2.0)
    
    # ================================================================================
    # 绘图
    # ================================================================================
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # 绘制瓦斯浓度曲线
    ax.plot(t, gas_data, 'k-', linewidth=0.8, label='瓦斯浓度')
    
    # 绘制阈值线
    ax.axhline(y=1.0, color='#FF7F0E', linestyle='--', linewidth=1.5, label='预警值(1.0%)')
    ax.axhline(y=1.5, color='#D62728', linestyle='--', linewidth=1.5, label='危险值(1.5%)')
    
    # 使用半透明背景色标注三种异常区域
    ax.axvspan(1200, 1210, alpha=0.25, color='red', label='模式A:突发尖峰')
    ax.axvspan(1800, 2100, alpha=0.25, color='orange', label='模式B:缓慢涌出')
    ax.axvspan(2700, 2750, alpha=0.25, color='purple', label='模式C:指数爆发')
    
    # 设置坐标轴标签
    ax.set_xlabel('时间(秒)', fontsize=12)
    ax.set_ylabel('瓦斯浓度(%)', fontsize=12)
    ax.set_title('图2：模拟瓦斯浓度数据(含三种典型异常模式)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 3600)
    ax.set_ylim(0, 2.2)
    
    plt.tight_layout()
    plt.savefig('fig2_raw_data.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("图2已生成: fig2_raw_data.png")


# ============================================================
# 图3：动态桶调整策略示意图
# ============================================================

def generate_figure3_bucket_adjustment():
    """图3：动态桶调整策略示意图"""
    np.random.seed(42)
    T = 3600
    
    # 生成数据
    base = 0.3 + 0.05 * np.random.randn(T)
    breath = 0.1 * np.sin(2 * np.pi * np.arange(T) / 300)
    anomalies = np.zeros(T)
    anomalies[1200:1210] = 0.8 * np.exp(-0.5 * np.arange(10))
    anomalies[1800:2100] = 0.005 * np.arange(300)
    anomalies[2700:2750] = 1.2 * (1 - np.exp(-0.1 * np.arange(50)))
    gas_data = np.clip(base + breath + anomalies, 0.1, 2.0)
    
    # 计算梯度
    G = np.abs(np.diff(gas_data))
    
    # 模拟动态桶大小
    B_base = 50
    alpha = 1.0
    B_min = 5
    B_max = 100
    
    # 将数据分成72个桶
    n_buckets = 72
    bucket_size = (T - 2) / (n_buckets - 2)
    dynamic_bucket_sizes = []
    
    for i in range(1, n_buckets):
        start = int((i - 1) * bucket_size) + 1
        end = int(i * bucket_size) + 1
        if end > len(G):
            end = len(G)
        if start < end:
            g_avg = np.mean(G[start:min(end, len(G))])
        else:
            g_avg = 0
        B_i = B_base / (1 + alpha * g_avg)
        B_i = np.clip(B_i, B_min, B_max)
        dynamic_bucket_sizes.append(B_i)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # 上图：瓦斯浓度
    axes[0].plot(np.arange(T), gas_data, 'b-', linewidth=0.8, label='瓦斯浓度')
    axes[0].axvspan(1200, 1210, alpha=0.2, color='red')
    axes[0].axvspan(1800, 2100, alpha=0.2, color='orange')
    axes[0].axvspan(2700, 2750, alpha=0.2, color='purple')
    axes[0].axhline(y=1.0, color='#FF7F0E', linestyle='--', linewidth=1.2)
    axes[0].set_ylabel('浓度(%)', fontsize=11)
    axes[0].set_title('图3(a)：瓦斯浓度随时间变化曲线', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)
    
    # 下图：动态桶大小
    bucket_centers = [(int((i) * bucket_size) + int((i+1) * bucket_size)) / 2 for i in range(len(dynamic_bucket_sizes))]
    axes[1].plot(bucket_centers[1:], dynamic_bucket_sizes[1:], 'r-', linewidth=1.5, label='动态桶大小')
    axes[1].axhline(y=B_max, color='green', linestyle='--', label=f'上限B_max={B_max}')
    axes[1].axhline(y=B_min, color='blue', linestyle='--', label=f'下限B_min={B_min}')
    
    # 标注异常区的桶变化
    axes[1].axvspan(1100, 1300, alpha=0.15, color='red', label='模式A区域')
    axes[1].axvspan(1700, 2200, alpha=0.15, color='orange', label='模式B区域')
    axes[1].axvspan(2600, 2800, alpha=0.15, color='purple', label='模式C区域')
    
    axes[1].set_xlabel('时间(秒)', fontsize=11)
    axes[1].set_ylabel('桶大小(B_i)', fontsize=11)
    axes[1].set_title('图3(b)：动态桶大小随时间变化(异常期桶变小,采样加密)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 120)
    
    plt.tight_layout()
    plt.savefig('fig3_bucket_adjustment.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("图3已生成: fig3_bucket_adjustment.png")


# ============================================================
# 图4：效果对比图
# ============================================================

def generate_figure4_comparison():
    """图4：传统LTTB与本发明效果对比"""
    np.random.seed(42)
    T = 3600
    
    # 生成数据
    base = 0.3 + 0.05 * np.random.randn(T)
    breath = 0.1 * np.sin(2 * np.pi * np.arange(T) / 300)
    anomalies = np.zeros(T)
    anomalies[1200:1210] = 0.8 * np.exp(-0.5 * np.arange(10))
    anomalies[1800:2100] = 0.005 * np.arange(300)
    anomalies[2700:2750] = 1.2 * (1 - np.exp(-0.1 * np.arange(50)))
    gas_data = np.clip(base + breath + anomalies, 0.1, 2.0)
    
    # 模拟采样（简化版）
    def simple_lttb_fixed(data, n_out):
        n = len(data)
        bucket = (n - 2) / (n_out - 2)
        indices = [0]
        for i in range(1, n_out - 1):
            start = int((i - 1) * bucket) + 1
            end = int(i * bucket) + 1
            indices.append((start + end) // 2)
        indices.append(n - 1)
        return np.array(indices)
    
    def simple_lttb_dynamic(data, n_out):
        n = len(data)
        G = np.abs(np.diff(data))
        indices = [0]
        
        # 动态桶
        buckets = []
        target_per_bucket = n / n_out
        for i in range(n_out - 2):
            pos = int((i + 1) * target_per_bucket)
            if pos < n - 1:
                g_avg = np.mean(G[max(0, pos-10):pos])
                bucket_size = max(3, min(80, int(50 / (1 + g_avg * 2))))
                buckets.append(bucket_size)
        
        pos = 0
        for b in buckets[:n_out-2]:
            pos = min(pos + b, n - 2)
            indices.append(pos)
        indices.append(n - 1)
        return np.array(indices)
    
    # 采样
    idx_fixed = simple_lttb_fixed(gas_data, 72)
    idx_dynamic = simple_lttb_dynamic(gas_data, 72)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 全局对比
    axes[0].plot(np.arange(T), gas_data, 'k-', linewidth=0.5, alpha=0.5, label='原始数据')
    axes[0].scatter(idx_fixed, gas_data[idx_fixed], c='blue', s=15, alpha=0.7, label=f'传统LTTB ({len(idx_fixed)}点)')
    axes[0].set_title('图4(a)：传统固定桶LTTB下采样结果', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('浓度(%)')
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(np.arange(T), gas_data, 'k-', linewidth=0.5, alpha=0.5, label='原始数据')
    axes[1].scatter(idx_dynamic, gas_data[idx_dynamic], c='red', s=15, alpha=0.7, label=f'本发明 ({len(idx_dynamic)}点)')
    axes[1].set_title('图4(b)：本发明动态LTTB下采样结果', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('浓度(%)')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)
    
    # 异常区放大
    axes[2].plot(np.arange(1190, 1220), gas_data[1190:1220], 'k-', linewidth=1, label='原始')
    fixed_in_A = idx_fixed[(idx_fixed >= 1190) & (idx_fixed < 1220)]
    dynamic_in_A = idx_dynamic[(idx_dynamic >= 1190) & (idx_dynamic < 1220)]
    axes[2].scatter(fixed_in_A, gas_data[fixed_in_A], c='blue', s=80, label=f'传统LTTB ({len(fixed_in_A)}点)')
    axes[2].scatter(dynamic_in_A, gas_data[dynamic_in_A], c='red', s=80, marker='*', label=f'本发明 ({len(dynamic_in_A)}点)')
    axes[2].set_title('图4(c)：模式A区域(突发尖峰)采样点对比放大', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('时间(秒)')
    axes[2].set_ylabel('浓度(%)')
    axes[2].legend(loc='upper right')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig4_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("图4已生成: fig4_comparison.png")


# ============================================================
# 图5：预警时间对比
# ============================================================

def generate_figure5_timing():
    """图5：预警时间对比示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # 时间线
    t = np.linspace(0, 100, 100)
    # 模拟瓦斯浓度上升
    concentration = 0.3 + 0.015 * t
    concentration = np.clip(concentration, 0.3, 1.8)
    
    ax.plot(t, concentration, 'b-', linewidth=2, label='瓦斯浓度变化')
    
    # 标注关键时间点
    T0 = 20  # 异常开始时刻
    T1 = 35  # 趋势预警时刻
    T2 = 55  # 阈值报警时刻
    
    ax.axvline(x=T0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=T1, color='green', linestyle='--', linewidth=2)
    ax.axvline(x=T2, color='red', linestyle='--', linewidth=2)
    
    ax.axhline(y=1.0, color='#FF7F0E', linestyle='-', linewidth=2, label='预警阈值(1.0%)')
    
    # 添加标注
    ax.annotate('T0\n异常开始', xy=(T0, 0.5), fontsize=10, ha='center', color='gray')
    ax.annotate('T1\n趋势预警\n(本发明)', xy=(T1, 0.9), fontsize=11, ha='center', color='green', fontweight='bold')
    ax.annotate('T2\n阈值报警\n(传统)', xy=(T2, 1.15), fontsize=11, ha='center', color='red', fontweight='bold')
    
    # 标注提前时间
    ax.annotate('', xy=(T1, 1.5), xytext=(T2, 1.5),
               arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.text((T1+T2)/2, 1.55, f'ΔT = T2-T1 = {T2-T1}秒', ha='center', fontsize=12, color='purple', fontweight='bold')
    
    ax.set_xlabel('时间(秒)', fontsize=12)
    ax.set_ylabel('瓦斯浓度(%)', fontsize=12)
    ax.set_title('图5：预警时间对比示意图(T1<T2, 本发明预警更早)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 2.0)
    
    plt.tight_layout()
    plt.savefig('fig5_timing.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("图5已生成: fig5_timing.png")


# ============================================================
# 图6：系统架构图
# ============================================================

def generate_figure6_architecture():
    """图6：系统架构示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 绘制设备框
    def draw_equipment(x, y, w, h, text, color='#E3F2FD'):
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1", 
                             facecolor=color, edgecolor='#1565C0', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
    
    def draw_arrow_simple(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))
    
    # 传感器
    draw_equipment(2, 8, 2, 1, '瓦斯传感器\n(1Hz-10Hz)', '#E3F2FD')
    draw_equipment(5, 8, 2, 1, '数据采集器', '#E3F2FD')
    draw_arrow_simple(3, 8, 4, 8)
    
    # 边缘计算网关
    draw_equipment(8, 8, 3, 1.2, '边缘计算网关\n(本发明算法)', '#FFF3E0')
    draw_arrow_simple(6, 8, 7.25, 8)
    
    # 数据流
    ax.text(5.5, 8.5, '数据流', fontsize=9, ha='center')
    
    # 处理模块
    draw_equipment(8, 6, 3, 0.8, '数据处理模块\n动二代调整+LTTB', '#E8F5E9')
    draw_equipment(8, 4.5, 3, 0.8, '预警分析模块\n双模式预警', '#E8F5E9')
    draw_arrow_simple(8, 7.4, 8, 6.8)
    draw_arrow_simple(8, 5.4, 8, 4.9)
    
    # 监控中心
    draw_equipment(11.5, 6, 2, 1, '监控中心', '#FCE4EC')
    draw_arrow_simple(9.5, 6, 10.5, 6)
    
    # 报警终端
    draw_equipment(11.5, 4, 2, 1, '报警终端', '#FCE4EC')
    draw_arrow_simple(9.5, 4.5, 10.5, 4.5)
    draw_arrow_simple(11.5, 5.5, 11.5, 4.5)
    
    # 云端/存储
    draw_equipment(2, 4, 2, 1, '本地存储', '#E3F2FD')
    draw_equipment(2, 2.5, 2, 1, '云端服务器\n(可选)', '#E3F2FD')
    draw_arrow_simple(2, 3.5, 2, 3)
    draw_arrow_simple(8, 3.8, 8, 3)
    draw_arrow_simple(7, 4, 3, 4)
    
    ax.set_title('图6：本发明系统架构示意图', fontsize=14, fontweight='bold', pad=20)
    
    # 添加说明文字
    ax.text(5.5, 2, '数据流 →', fontsize=10, color='#1565C0')
    ax.text(8, 3, '↓ 控制流', fontsize=10, color='#1565C0')
    
    plt.tight_layout()
    plt.savefig('fig6_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("图6已生成: fig6_architecture.png")


# ============================================================
# 图7：抗噪机制效果图
# ============================================================

def generate_figure7_denoise():
    """图7：抗噪机制效果图"""
    np.random.seed(42)
    T = 200
    t = np.arange(T)
    
    # 正常数据 + 异常突增 + 噪声
    base = 0.3 + 0.02 * np.random.randn(T)
    base[50:60] = 0.3 + 0.8 * np.exp(-0.5 * np.arange(10))  # 突增
    
    # 添加噪声
    noise = np.zeros(T)
    noise[np.random.choice(T, 8, replace=False)] = 0.3 * np.random.randn(8)
    
    raw_data = base + noise
    raw_data = np.clip(raw_data, 0.1, 2.0)
    
    # 中值滤波
    from scipy.signal import medfilt
    filtered_data = medfilt(raw_data, kernel_size=3)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    
    # 带噪声的原始数据
    axes[0].plot(t, raw_data, 'r-', linewidth=1, label='带噪声的原始数据')
    axes[0].scatter(np.where(np.abs(noise) > 0.1)[0], raw_data[np.abs(noise) > 0.1], 
                   c='red', s=80, marker='x', label='噪声点(毛刺)')
    axes[0].set_title('图7(a)：含传感器毛刺的原始数据', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('浓度(%)')
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)
    
    # 滤波后
    axes[1].plot(t, raw_data, 'r-', linewidth=0.5, alpha=0.4, label='原始数据')
    axes[1].plot(t, filtered_data, 'b-', linewidth=1.5, label='中值滤波后(保留边缘)')
    axes[1].set_title('图7(b)：中值滤波去噪后的数据(保留真实突增边缘)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('浓度(%)')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)
    
    # 对比
    axes[2].plot(t, raw_data, 'r-', linewidth=0.5, alpha=0.3, label='原始')
    axes[2].plot(t, filtered_data, 'g-', linewidth=2, label='中值滤波')
    # 均值滤波对比
    from scipy.ndimage import uniform_filter1d
    mean_filtered = uniform_filter1d(raw_data, size=3)
    axes[2].plot(t, mean_filtered, 'b--', linewidth=1.5, alpha=0.7, label='均值滤波(会模糊边缘)')
    axes[2].set_title('图7(c)：中值滤波 vs 均值滤波效果对比', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('时间(秒)', fontsize=11)
    axes[2].set_ylabel('浓度(%)')
    axes[2].legend(loc='upper right')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig7_denoise.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("图7已生成: fig7_denoise.png")


# ============================================================
# 主程序：生成所有附图
# ============================================================

def main():
    print("="*60)
    print("开始生成专利附图...")
    print("="*60)
    
    generate_figure1_flowchart()
    generate_figure2_raw_data()
    generate_figure3_bucket_adjustment()
    generate_figure4_comparison()
    generate_figure5_timing()
    generate_figure6_architecture()
    generate_figure7_denoise()
    
    print("\n" + "="*60)
    print("附图生成完成！共7张图片")
    print("="*60)
    print("生成的文件列表：")
    print("  1. fig1_flowchart.png - 方法总体流程图")
    print("  2. fig2_raw_data.png - 模拟瓦斯数据")
    print("  3. fig3_bucket_adjustment.png - 动态桶调整策略")
    print("  4. fig4_comparison.png - 效果对比图")
    print("  5. fig5_timing.png - 预警时间对比")
    print("  6. fig6_architecture.png - 系统架构图")
    print("  7. fig7_denoise.png - 抗噪机制效果")


if __name__ == "__main__":
    main()