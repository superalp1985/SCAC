import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# ==================== 实验7：信息增益上界 ====================
print("="*60)
print("实验7：信息增益上界 (Theorem 16: Information Gain Upper Bound)")
print("验证：单次反馈的信息增益 ≤ log|可行集|")
print("="*60)

np.random.seed(42)

# 参数设置
n_trials = 500           # 重复实验次数
feasible_sizes = [10, 20, 50, 100, 200, 500]  # 不同可行集大小
noise_level = 0.1        # 反馈噪声（模拟不完全确定）

# 存储信息增益
mean_info_gains = []
std_info_gains = []
theoretical_bounds = [np.log(N) for N in feasible_sizes]

for N in feasible_sizes:
    info_gains = []
    for _ in range(n_trials):
        # 初始分布：假设均匀分布（熵最大）
        p_prior = np.ones(N) / N
        H_prior = entropy(p_prior)
        
        # 模拟反馈：假设反馈能指向某个子集（大小为N_eff）
        # 为了保守，假设反馈后分布集中在K个可行解上
        K = max(1, int(N * 0.3))  # 假设反馈能将候选集缩小到30%
        p_posterior = np.zeros(N)
        # 随机选K个索引，分配均匀概率
        indices = np.random.choice(N, K, replace=False)
        p_posterior[indices] = 1.0 / K
        # 添加少量噪声（使不完全确定）
        p_posterior += np.random.uniform(0, noise_level / N, N)
        p_posterior /= p_posterior.sum()
        H_posterior = entropy(p_posterior)
        
        info_gain = H_prior - H_posterior
        info_gains.append(info_gain)
    
    mean_info_gains.append(np.mean(info_gains))
    std_info_gains.append(np.std(info_gains))

# 打印统计信息
print("\n【实验结果】")
print(f"{'可行集大小 N':<15}{'平均信息增益':<18}{'标准差':<12}{'理论界 logN':<12}{'是否 ≤ 理论界'}")
for i, N in enumerate(feasible_sizes):
    actual = mean_info_gains[i]
    theory = theoretical_bounds[i]
    ok = "✅" if actual <= theory else "❌"
    print(f"{N:<15}{actual:<18.4f}{std_info_gains[i]:<12.4f}{theory:<12.4f}{ok}")

# ==================== 可视化 ====================
plt.figure(figsize=(15, 5))

# 图1：信息增益 vs 可行集大小
plt.subplot(1, 3, 1)
plt.errorbar(feasible_sizes, mean_info_gains, yerr=std_info_gains, 
             fmt='bo-', capsize=5, capthick=2, label='Actual Info Gain')
plt.plot(feasible_sizes, theoretical_bounds, 'r--', linewidth=2, label='Theoretical bound log(N)')
plt.xlabel('Feasible set size N')
plt.ylabel('Information gain (nats)')
plt.title('Theorem 16: Information Gain Upper Bound')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.yscale('log')

# 图2：实际增益与理论界的比例
plt.subplot(1, 3, 2)
ratios = [mean_info_gains[i] / theoretical_bounds[i] for i in range(len(feasible_sizes))]
plt.plot(feasible_sizes, ratios, 'go-', linewidth=2, markersize=8)
plt.axhline(y=1, color='red', linestyle='--', label='Ratio = 1')
plt.xlabel('Feasible set size N')
plt.ylabel('Actual / Theory')
plt.title('Ratio of Actual Gain to Theoretical Bound')
plt.grid(True, alpha=0.3)
plt.xscale('log')

# 图3：不同噪声水平下的信息增益（固定N=100）
plt.subplot(1, 3, 3)
noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
N_fixed = 100
info_vs_noise = []
for noise in noise_levels:
    gains = []
    for _ in range(200):
        p_prior = np.ones(N_fixed) / N_fixed
        H_prior = entropy(p_prior)
        K = max(1, int(N_fixed * 0.3))
        p_posterior = np.zeros(N_fixed)
        indices = np.random.choice(N_fixed, K, replace=False)
        p_posterior[indices] = 1.0 / K
        p_posterior += np.random.uniform(0, noise / N_fixed, N_fixed)
        p_posterior /= p_posterior.sum()
        H_posterior = entropy(p_posterior)
        gains.append(H_prior - H_posterior)
    info_vs_noise.append(np.mean(gains))

plt.plot(noise_levels, info_vs_noise, 'mo-', linewidth=2, markersize=8)
plt.axhline(y=np.log(N_fixed), color='red', linestyle='--', label=f'log({N_fixed}) = {np.log(N_fixed):.2f}')
plt.xlabel('Noise level')
plt.ylabel('Information gain')
plt.title('Info Gain vs Noise (N=100)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('experiment7_info_gain.png', dpi=150)
plt.show()

# ==================== 定理验证 ====================
print("\n" + "="*60)
print("【定理16验证结果】")

# 验证1：所有点是否都 ≤ 理论界
all_ok = all(mean_info_gains[i] <= theoretical_bounds[i] for i in range(len(feasible_sizes)))
if all_ok:
    print("✅ 所有可行集下，平均信息增益 ≤ logN，上界成立 ✓")
else:
    print("❌ 部分点超出上界")

# 验证2：平均比例
avg_ratio = np.mean(ratios)
if avg_ratio < 1:
    print(f"✅ 平均比例 {avg_ratio:.2f} < 1，理论保守合理 ✓")
else:
    print(f"⚠️ 平均比例 {avg_ratio:.2f} 接近1，需要检查")

# 验证3：噪声影响
if info_vs_noise[-1] < info_vs_noise[0]:
    print("✅ 噪声越大，信息增益越小，符合直觉 ✓")
else:
    print("⚠️ 噪声影响不明显")

print("\n" + "="*60)
print("✅ 实验完成！结果已保存为 experiment7_info_gain.png")
print("="*60)

input("\n按回车键退出...")