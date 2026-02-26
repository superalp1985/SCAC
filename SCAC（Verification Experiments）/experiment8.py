import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# ==================== 实验8：贝叶斯更新 ====================
print("="*60)
print("实验8：贝叶斯更新 (Theorem 21: Bayesian Update)")
print("验证：后验分布 ∝ 先验 × 似然，熵单调递减")
print("="*60)

np.random.seed(42)

# 参数设置
n_states = 10            # 状态数
n_trials = 1000          # 重复实验次数
n_iters = 5              # 每轮更新次数

# 存储熵的历史
entropy_history = np.zeros((n_trials, n_iters+1))

for trial in range(n_trials):
    # 随机生成先验分布
    prior = np.random.dirichlet(np.ones(n_states))
    
    # 随机生成真实状态
    true_state = np.random.randint(0, n_states)
    
    # 似然函数：观测到正确状态的概率高
    likelihood = np.ones(n_states) * 0.1
    likelihood[true_state] = 0.9
    
    # 记录初始熵
    current = prior.copy()
    entropy_history[trial, 0] = entropy(current)
    
    for t in range(1, n_iters+1):
        # 贝叶斯更新：后验 ∝ 先验 × 似然
        posterior = current * likelihood
        posterior /= posterior.sum()
        
        # 记录熵
        entropy_history[trial, t] = entropy(posterior)
        
        # 下一次的先验用这次的后验
        current = posterior

# 计算统计量
mean_entropy = np.mean(entropy_history, axis=0)
std_entropy = np.std(entropy_history, axis=0)

print("\n【实验结果】")
print(f"{'迭代次数':<12}{'平均熵':<15}{'标准差':<15}")
for t in range(n_iters+1):
    print(f"{t:<12}{mean_entropy[t]:<15.4f}{std_entropy[t]:<15.4f}")

# ==================== 可视化 ====================
plt.figure(figsize=(12, 4))

# 图1：熵演化
plt.subplot(1, 3, 1)
plt.errorbar(range(n_iters+1), mean_entropy, yerr=std_entropy, 
             fmt='o-', capsize=5, capthick=2, color='blue')
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.title('Theorem 21: Bayesian Update\n(Entropy Decreases)')
plt.grid(True, alpha=0.3)

# 图2：单次实验示例
plt.subplot(1, 3, 2)
example_idx = 0
plt.plot(range(n_iters+1), entropy_history[example_idx], 'ro-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Entropy')
plt.title('Single Trial Example')
plt.grid(True, alpha=0.3)

# 图3：最终分布 vs 先验
plt.subplot(1, 3, 3)
plt.bar(range(n_states), prior, alpha=0.7, label='Prior')
plt.bar(range(n_states), current, alpha=0.7, label='Final Posterior')
plt.xlabel('State')
plt.ylabel('Probability')
plt.title('Prior vs Posterior')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiment8_bayesian_update.png', dpi=150)
plt.show()

# ==================== 定理验证 ====================
print("\n" + "="*60)
print("【定理21验证结果】")

# 验证1：熵是否单调递减
if np.all(np.diff(mean_entropy) < 0):
    print("✅ 熵单调递减，贝叶斯更新成立 ✓")
else:
    print("❌ 熵未单调递减")

# 验证2：最终分布是否集中在真实状态
final_state = np.argmax(current)
if final_state == true_state:
    print("✅ 后验分布集中在真实状态 ✓")
else:
    print("⚠️ 后验分布可能未完全集中")

print("="*60)
input("\n按回车键退出...")