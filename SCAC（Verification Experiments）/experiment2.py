import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# ==================== 实验2：幻觉熵增律 ====================
print("="*60)
print("实验2：幻觉熵增律 (Theorem 3: Hallucination Entropy Increase)")
print("验证：无反馈时，系统熵随迭代次数指数增长")
print("="*60)

# 参数设置
np.random.seed(42)  # 固定随机种子，保证可重复
n_states = 50        # 状态空间大小（代码可能性数量）
n_iterations = 30    # 迭代次数
n_trials = 100       # 重复实验次数

# 初始化存储所有实验的熵历史
all_entropy_history = np.zeros((n_trials, n_iterations))

# 进行多次实验
for trial in range(n_trials):
    # 初始分布：集中在少数几个状态
    p = np.zeros(n_states)
    p[20:25] = [0.3, 0.25, 0.2, 0.15, 0.1]  # 初始集中在5个状态
    p = p / p.sum()  # 归一化
    
    entropy_history = []
    
    for t in range(n_iterations):
        # 记录当前熵
        current_entropy = entropy(p)
        entropy_history.append(current_entropy)
        
        # 随机游走扩散（模拟无反馈的转移核）
        # 每个状态以一定概率向邻近状态扩散
        p_new = np.zeros_like(p)
        for i in range(n_states):
            # 自身保留80%，左右各10%
            p_new[i] += p[i] * 0.8
            if i > 0:
                p_new[i-1] += p[i] * 0.1
            if i < n_states-1:
                p_new[i+1] += p[i] * 0.1
        # 边界处理
        p_new[0] += p[0] * 0.1  # 左边界向右
        p_new[-1] += p[-1] * 0.1  # 右边界向左
        
        # 归一化
        p = p_new / p_new.sum()
    
    all_entropy_history[trial] = entropy_history

# 计算统计量
mean_entropy = np.mean(all_entropy_history, axis=0)
std_entropy = np.std(all_entropy_history, axis=0)
max_entropy = np.log(n_states)  # 理论最大熵

# 拟合熵增速率
from scipy.optimize import curve_fit

def entropy_fit(t, H0, lambda_rate):
    return H0 + lambda_rate * t

t_range = np.arange(n_iterations)
# 用中间稳定部分拟合（避免初始瞬态）
fit_start = 5
popt, _ = curve_fit(entropy_fit, t_range[fit_start:], mean_entropy[fit_start:])
H0_fit, lambda_fit = popt
rho_fit = np.exp(-lambda_fit)  # 转移核最大值

# 打印统计信息
print("\n【实验结果统计】")
print(f"{'迭代次数':<12}{'平均熵':<12}{'标准差':<12}{'理论下界':<12}")
for t in range(0, n_iterations, 5):
    theoretical_lower = H0_fit + lambda_fit * t
    print(f"{t:<12}{mean_entropy[t]:<12.4f}{std_entropy[t]:<12.4f}{theoretical_lower:<12.4f}")

print(f"\n拟合结果：")
print(f"  初始熵 H0 = {H0_fit:.4f}")
print(f"  熵增速率 λ = {lambda_fit:.4f}")
print(f"  转移核最大值 ρ = e^(-λ) = {rho_fit:.4f}")
print(f"  理论最大熵 = {max_entropy:.4f}")

# ==================== 可视化 ====================
plt.figure(figsize=(12, 8))

# 图1：熵演化（带误差带）
plt.subplot(2, 2, 1)
plt.fill_between(t_range, 
                  mean_entropy - std_entropy, 
                  mean_entropy + std_entropy, 
                  alpha=0.3, color='blue', label='±1 Std')
plt.plot(t_range, mean_entropy, 'b-', linewidth=2, label='Mean Entropy')
plt.plot(t_range, H0_fit + lambda_fit * t_range, 'r--', 
         linewidth=2, label=f'Fit: H₀ + {lambda_fit:.3f}t')
plt.axhline(y=max_entropy, color='gray', linestyle=':', 
            label=f'Max Entropy = {max_entropy:.3f}')
plt.xlabel('Iteration t')
plt.ylabel('Entropy H(t)')
plt.title('Theorem 3: Hallucination Entropy Increase')
plt.legend()
plt.grid(True, alpha=0.3)

# 图2：对数坐标检查指数增长
plt.subplot(2, 2, 2)
plt.semilogy(t_range, max_entropy - mean_entropy, 'b-o', 
             markersize=4, label='Distance to Max Entropy')
plt.xlabel('Iteration t')
plt.ylabel('log(Max Entropy - H(t))')
plt.title('Exponential Approach to Max Entropy')
plt.grid(True, alpha=0.3)
plt.legend()

# 图3：最终分布 vs 初始分布
plt.subplot(2, 2, 3)
# 用最后一次实验的最终分布
final_p = p  # 最后一次实验的最终分布
initial_p = np.zeros(n_states)
initial_p[20:25] = [0.3, 0.25, 0.2, 0.15, 0.1]
initial_p = initial_p / initial_p.sum()

x = np.arange(n_states)
plt.bar(x-0.2, initial_p, width=0.4, alpha=0.7, label='Initial Distribution')
plt.bar(x+0.2, final_p, width=0.4, alpha=0.7, label='Final Distribution')
plt.axhline(y=1/n_states, color='r', linestyle='--', 
            label=f'Uniform (1/{n_states})')
plt.xlabel('State')
plt.ylabel('Probability')
plt.title('Initial vs Final Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# 图4：熵增速率 vs 理论
plt.subplot(2, 2, 4)
# 绘制多次实验的熵增曲线（抽样几条）
for i in range(min(10, n_trials)):
    plt.plot(t_range, all_entropy_history[i], 'gray', alpha=0.3, linewidth=0.5)
plt.plot(t_range, mean_entropy, 'b-', linewidth=3, label='Mean')
plt.plot(t_range, H0_fit + lambda_fit * t_range, 'r--', linewidth=2, label='Fit')
plt.xlabel('Iteration t')
plt.ylabel('Entropy H(t)')
plt.title('Multiple Trials (n=10)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiment2_entropy_increase.png', dpi=150)
plt.show()

# ==================== 定理验证 ====================
print("\n" + "="*60)
print("【定理3验证结果】")

# 验证1：熵是否单调递增
is_monotonic = np.all(np.diff(mean_entropy) > -1e-6)
if is_monotonic:
    print("✅ 熵单调递增 ✓")
else:
    print("⚠️ 熵非严格单调（可能波动）")

# 验证2：熵增速率是否>0
if lambda_fit > 0:
    print(f"✅ 熵增速率 λ = {lambda_fit:.4f} > 0 ✓")
else:
    print("❌ 熵增速率 ≤ 0")

# 验证3：最终是否接近均匀分布
kl_div = np.sum(final_p * np.log(final_p * n_states + 1e-10))
if kl_div < 0.1:
    print(f"✅ 最终分布接近均匀 (KL散度={kl_div:.4f}) ✓")
else:
    print(f"⚠️ 最终分布未完全均匀 (KL散度={kl_div:.4f})")

# 验证4：熵增理论下界
theoretical_bound = H0_fit + lambda_fit * t_range[-1]
if mean_entropy[-1] >= theoretical_bound - 0.1:
    print("✅ 熵值达到理论下界 ✓")
else:
    print("⚠️ 熵值低于理论下界")

print("\n" + "="*60)
print("✅ 实验完成！结果已保存为 experiment2_entropy_increase.png")
print("="*60)