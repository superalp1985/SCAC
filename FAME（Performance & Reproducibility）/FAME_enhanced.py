import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

# ==================== FAME 增强版 v2 ====================
print("="*60)
print("FAME 增强版 v2：带L1正则的熵最小化")
print("目标：权重稀疏 + 熵显著下降")
print("="*60)

np.random.seed(20260226)

# 生成数据：10个信号特征 + 90个噪声
n_samples = 2000
n_features = 100
n_signal = 10

X_signal = np.random.randn(n_samples, n_signal) * 2
y_true = np.sum(X_signal, axis=1)
X_noise = np.random.randn(n_samples, n_features - n_signal) * 5
X = np.hstack([X_signal, X_noise])

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==================== FAME 核心类 ====================
class FAME:
    def __init__(self, n_features, lr=0.5, l1_lambda=0.01):
        self.w = np.ones(n_features) / n_features
        self.lr = lr
        self.l1 = l1_lambda
        
    def forward(self, X):
        return X @ self.w
    
    def compute_entropy_gradient(self, X, y, bins=50):
        """直接计算熵对权重的梯度"""
        # 用直方图估计概率分布
        hist, edges = np.histogram(y, bins=bins)
        probs = hist / np.sum(hist)
        
        # 计算每个样本所属的bin
        indices = np.digitize(y, edges[:-1]) - 1
        indices = np.clip(indices, 0, bins-1)
        
        # 熵对权重的梯度 = 每个样本对权重的贡献
        grad = np.zeros_like(self.w)
        for i in range(n_samples):
            bin_idx = indices[i]
            if probs[bin_idx] > 0:
                # 熵的梯度公式：dH/dy = -log(p(y)) * dp/dy
                # 这里简化：用当前bin的概率倒数
                grad += X[i] * (1.0 / (probs[bin_idx] + 1e-10))
        
        return grad / n_samples
    
    def update(self, X):
        y = self.forward(X)
        
        # 熵梯度
        entropy_grad = self.compute_entropy_gradient(X, y)
        
        # L1正则梯度
        l1_grad = self.l1 * np.sign(self.w)
        
        # 总梯度
        grad = entropy_grad + l1_grad
        grad = grad / (np.linalg.norm(grad) + 1e-10)
        
        # 更新权重
        self.w -= self.lr * grad
        
        # 投影：非负且和为1
        self.w = np.maximum(self.w, 0)
        self.w /= self.w.sum()
        
        return entropy(y)

# ==================== 训练 ====================
fame = FAME(n_features, lr=0.5, l1_lambda=0.01)
entropy_history = []
weight_history = []

n_epochs = 100
for epoch in range(n_epochs):
    ent = fame.update(X)
    entropy_history.append(ent)
    weight_history.append(fame.w.copy())
    
    if (epoch+1) % 20 == 0:
        # 计算当前信噪比
        signal_weight = np.sum(fame.w[:n_signal])
        noise_weight = np.sum(fame.w[n_signal:])
        snr = signal_weight / (noise_weight + 1e-10)
        print(f"Epoch {epoch+1}: 熵={ent:.4f}, SNR={snr:.2f}")

# 基线（均匀权重）
y_base = X @ (np.ones(n_features) / n_features)
ent_base = entropy(np.histogram(y_base, bins=50)[0] + 1e-10)

print("\n" + "="*60)
print(f"基线熵: {ent_base:.4f}")
print(f"最终熵: {entropy_history[-1]:.4f}")
print(f"熵减: {ent_base - entropy_history[-1]:.4f}")

# ==================== 可视化 ====================
plt.figure(figsize=(15, 10))

# 图1：熵下降
plt.subplot(2, 2, 1)
plt.plot(entropy_history, 'b-', linewidth=2)
plt.axhline(y=ent_base, color='r', linestyle='--', label=f'Baseline: {ent_base:.4f}')
plt.xlabel('Epoch')
plt.ylabel('Entropy')
plt.title('FAME: Entropy Minimization')
plt.legend()
plt.grid(True, alpha=0.3)

# 图2：权重演化（前20个）
plt.subplot(2, 2, 2)
weight_history = np.array(weight_history)
for i in range(20):
    plt.plot(weight_history[:, i], label=f'Feat {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Weight')
plt.title('FAME: Weight Evolution (Top 20)')
plt.grid(True, alpha=0.3)

# 图3：最终权重分布
plt.subplot(2, 2, 3)
plt.bar(range(n_features), fame.w)
plt.axvline(x=n_signal-0.5, color='r', linestyle='--', label='Signal/Noise boundary')
plt.xlabel('Feature index')
plt.ylabel('Final weight')
plt.title('FAME: Learned Feature Weights')
plt.legend()
plt.grid(True, alpha=0.3)

# 图4：信噪比
plt.subplot(2, 2, 4)
signal_weight = np.sum(fame.w[:n_signal])
noise_weight = np.sum(fame.w[n_signal:])
snr_fame = signal_weight / (noise_weight + 1e-10)
snr_base = n_signal / (n_features - n_signal)

plt.bar(['Baseline', 'FAME'], [snr_base, snr_fame])
plt.ylabel('Signal/Noise Ratio')
plt.title('FAME: SNR Enhancement')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('FAME_v2_results.png', dpi=150)
plt.show()