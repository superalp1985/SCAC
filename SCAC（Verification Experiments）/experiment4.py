import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==================== 实验4：压缩映射定理 ====================
print("="*60)
print("实验4：压缩映射定理 (Theorem 9: Contraction Mapping)")
print("验证：有物理反馈时，误差指数收敛")
print("="*60)

# 设置随机种子
np.random.seed(42)

# 参数设置
n_iters = 20             # 迭代次数
n_trials = 100           # 重复实验次数
target = 50              # 目标意图（理想代码位置）
initial_distance = 30    # 初始距离
kappa_true = 0.65        # 理论收缩因子

# 存储每次实验的距离历史
distances = np.zeros((n_trials, n_iters))

for trial in range(n_trials):
    # 初始位置
    pos = target - initial_distance
    if trial % 2 == 0:
        pos = target + initial_distance  # 一半从左边，一半从右边
    
    for t in range(n_iters):
        # 记录当前距离
        dist = abs(pos - target)
        distances[trial, t] = dist
        
        # 压缩映射：向目标收缩
        # 理论：d_{t+1} = κ * d_t
        pos = target + (pos - target) * kappa_true
        
        # 添加微小噪声（模拟现实中的不确定性）
        noise = np.random.normal(0, 0.5)
        pos += noise

# 计算统计量
mean_dist = np.mean(distances, axis=0)
std_dist = np.std(distances, axis=0)
median_dist = np.median(distances, axis=0)

# 拟合实际收缩因子
def exp_decay(t, d0, kappa):
    return d0 * (kappa ** t)

t_range = np.arange(n_iters)
# 用前10次拟合
popt, _ = curve_fit(exp_decay, t_range[:10], mean_dist[:10], p0=[initial_distance, 0.7])
d0_fit, kappa_fit = popt

# 打印统计信息
print("\n【实验结果】")
print(f"{'迭代次数':<12}{'平均距离':<15}{'标准差':<15}{'理论距离 (κ=0.65)':<20}")
for t in range(0, n_iters, 3):
    theoretical = initial_distance * (kappa_true ** t)
    print(f"{t:<12}{mean_dist[t]:<15.4f}{std_dist[t]:<15.4f}{theoretical:<20.4f}")

print(f"\n拟合结果：")
print(f"  实际收缩因子 κ_fit = {kappa_fit:.4f}")
print(f"  理论收缩因子 κ_true = {kappa_true:.4f}")
print(f"  误差 = {abs(kappa_fit - kappa_true):.4f}")

# ==================== 可视化 ====================
plt.figure(figsize=(15, 10))

# 图1：距离演化（线性坐标）
plt.subplot(2, 2, 1)
plt.fill_between(t_range, 
                 mean_dist - std_dist, 
                 mean_dist + std_dist, 
                 alpha=0.3, color='blue', label='±1 Std')
plt.plot(t_range, mean_dist, 'b-o', linewidth=2, markersize=6, label='Mean distance')
plt.plot(t_range, initial_distance * (kappa_true ** t_range), 
         'r--', linewidth=2, label=f'Theoretical: d₀·κ^t, κ={kappa_true}')
plt.xlabel('Iteration t')
plt.ylabel('Distance |l_t - c*|')
plt.title('Theorem 9: Exponential Convergence (Linear Scale)')
plt.legend()
plt.grid(True, alpha=0.3)

# 图2：距离演化（对数坐标）——验证指数性
plt.subplot(2, 2, 2)
plt.semilogy(t_range, mean_dist, 'b-o', linewidth=2, markersize=6, label='Mean distance')
plt.semilogy(t_range, initial_distance * (kappa_true ** t_range), 
             'r--', linewidth=2, label=f'Theoretical: κ={kappa_true}')
# 绘制多次实验的样本轨迹（抽样10条）
for i in range(min(10, n_trials)):
    plt.semilogy(t_range, distances[i], 'gray', alpha=0.3, linewidth=0.8)
plt.xlabel('Iteration t')
plt.ylabel('Distance (log scale)')
plt.title('Exponential Convergence (Log Scale)')
plt.legend()
plt.grid(True, alpha=0.3)

# 图3：残差分析
plt.subplot(2, 2, 3)
residual = mean_dist - initial_distance * (kappa_true ** t_range)
plt.bar(t_range, residual, width=0.6, alpha=0.7, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Iteration t')
plt.ylabel('Residual (Mean - Theoretical)')
plt.title('Residual Analysis')
plt.grid(True, alpha=0.3)

# 图4：κ值的分布（验证收敛性）
plt.subplot(2, 2, 4)
# 对每次实验单独拟合κ
kappa_estimates = []
for trial in range(n_trials):
    try:
        popt_trial, _ = curve_fit(exp_decay, t_range[:10], distances[trial, :10], p0=[initial_distance, 0.7])
        kappa_estimates.append(popt_trial[1])
    except:
        pass

plt.hist(kappa_estimates, bins=20, alpha=0.7, color='purple', edgecolor='black')
plt.axvline(x=kappa_true, color='red', linestyle='--', linewidth=2, label=f'True κ={kappa_true}')
plt.axvline(x=np.mean(kappa_estimates), color='blue', linestyle=':', linewidth=2, label=f'Mean κ={np.mean(kappa_estimates):.4f}')
plt.xlabel('Estimated κ')
plt.ylabel('Frequency')
plt.title('Distribution of Estimated Contraction Factor')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiment4_contraction_mapping.png', dpi=150)
plt.show()

# ==================== 定理验证 ====================
print("\n" + "="*60)
print("【定理9验证结果】")

# 验证1：是否指数衰减（对数坐标直线）
# 计算对数坐标下的线性相关系数
log_mean = np.log(mean_dist[:15] + 1e-10)  # 避免log(0)
corr = np.corrcoef(t_range[:15], log_mean)[0, 1]
if abs(corr) > 0.99:
    print(f"✅ 对数坐标线性相关系数 = {corr:.4f} > 0.99，指数衰减成立 ✓")
else:
    print(f"⚠️ 线性相关系数 = {corr:.4f}，可能不是完美指数")

# 验证2：收缩因子 < 1
if kappa_fit < 1:
    print(f"✅ 收缩因子 κ = {kappa_fit:.4f} < 1，压缩映射成立 ✓")
else:
    print(f"❌ 收缩因子 κ = {kappa_fit:.4f} ≥ 1，压缩映射不成立")

# 验证3：与理论值一致
if abs(kappa_fit - kappa_true) < 0.05:
    print(f"✅ 拟合κ与理论κ误差 < 0.05，一致 ✓")
else:
    print(f"⚠️ 拟合κ与理论κ误差较大")

# 验证4：残差随机（无明显趋势）
if np.std(residual[-5:]) < 0.5:
    print("✅ 残差稳定，收敛完成 ✓")
else:
    print("⚠️ 残差仍有波动，可能未完全收敛")

print("\n" + "="*60)
print("✅ 实验完成！结果已保存为 experiment4_contraction_mapping.png")
print("="*60)

# 增加暂停，防止闪退
input("\n按回车键退出...")