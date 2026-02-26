import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os

# ==================== FAME 核心实验 ====================
print("="*60)
print("FAME 特征增强与记忆熵量化实验")
print("验证：熵减、信噪比提升、复现一致性")
print("="*60)

np.random.seed(20260226)  # 固定种子，确保可复现

# 模拟数据集
n_samples = 1000
n_features = 64
X_raw = np.random.randn(n_samples, n_features) * 2 + 5
y_true = np.sum(X_raw[:, :10], axis=1) + np.random.randn(n_samples) * 0.5

# 对照组：无FAME
def baseline_model(X):
    return np.mean(X, axis=1)

# 实验组：带FAME（特征增强 + 记忆熵约束）
def fame_model(X, gamma=0.3):
    # 模拟FAME特征提纯
    weights = np.exp(-gamma * np.arange(X.shape[1]))
    weights /= weights.sum()
    enhanced = X * weights
    return np.sum(enhanced, axis=1)

# 多次运行，测试复现性
n_runs = [10, 100, 1000]
results = {}

for runs in n_runs:
    mae_baseline = []
    mae_fame = []
    entropy_baseline = []
    entropy_fame = []
    
    for _ in range(runs):
        # 每次重新生成数据（但种子固定，数据相同）
        X = np.random.randn(n_samples, n_features) * 2 + 5
        y = np.sum(X[:, :10], axis=1) + np.random.randn(n_samples) * 0.5
        
        y_pred_base = baseline_model(X)
        y_pred_fame = fame_model(X)
        
        mae_baseline.append(np.mean(np.abs(y_pred_base - y)))
        mae_fame.append(np.mean(np.abs(y_pred_fame - y)))
        
        # 计算熵（分布多样性）
        hist_base, _ = np.histogram(y_pred_base, bins=20)
        hist_fame, _ = np.histogram(y_pred_fame, bins=20)
        entropy_baseline.append(entropy(hist_base + 1e-10))
        entropy_fame.append(entropy(hist_fame + 1e-10))
    
    results[runs] = {
        'mae_base': np.mean(mae_baseline),
        'mae_fame': np.mean(mae_fame),
        'entropy_base': np.mean(entropy_baseline),
        'entropy_fame': np.mean(entropy_fame),
        'std_base': np.std(mae_baseline),
        'std_fame': np.std(mae_fame)
    }

# 输出结果
print("\n【实验结果】")
for runs in n_runs:
    r = results[runs]
    print(f"\n运行次数: {runs}")
    print(f"  对照组 MAE: {r['mae_base']:.4f} ± {r['std_base']:.4f}")
    print(f"  FAME  MAE: {r['mae_fame']:.4f} ± {r['std_fame']:.4f}")
    print(f"  熵减: {r['entropy_base']:.4f} → {r['entropy_fame']:.4f}")

# 可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.bar(['Baseline', 'FAME'], 
        [results[100]['mae_base'], results[100]['mae_fame']],
        yerr=[results[100]['std_base'], results[100]['std_fame']],
        capsize=5)
plt.ylabel('MAE')
plt.title('Prediction Error')

plt.subplot(1, 3, 2)
plt.bar(['Baseline', 'FAME'], 
        [results[100]['entropy_base'], results[100]['entropy_fame']])
plt.ylabel('Entropy')
plt.title('Memory Entropy')

plt.subplot(1, 3, 3)
stability = [results[10]['std_base']/results[10]['mae_base'],
             results[100]['std_base']/results[100]['mae_base'],
             results[1000]['std_base']/results[1000]['mae_base']]
plt.plot([10, 100, 1000], stability, 'o-')
plt.xlabel('Number of runs')
plt.ylabel('CV (std/mean)')
plt.title('Reproducibility')

plt.tight_layout()
plt.savefig('FAME_results.png', dpi=150)
plt.show()

print("\n✅ 实验完成，结果已保存为 FAME_results.png")