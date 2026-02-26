import numpy as np
import matplotlib.pyplot as plt

# ==================== SCAC-OPT 最优性验证实验 ====================
print("="*60)
print("SCAC-OPT：最优性、稳定性、收敛效率验证")
print("实验：不同参数下SCAC是否总能收敛到稳定最优解")
print("="*60)

np.random.seed(20260226)

# 基础设置
target = 100.0
n_iters = 30
n_trials = 10

# 待测试的参数组合（学习率/收缩因子κ）
params = [0.5, 0.6, 0.7, 0.8, 0.9]
results = {}

for kappa in params:
    final_errors = []
    steps_to_converge = []
    
    for _ in range(n_trials):
        pos = 0.0
        for t in range(n_iters):
            pos = target + (pos - target) * kappa
            pos += np.random.normal(0, 1.0)
            if abs(pos - target) < 1.0:
                steps_to_converge.append(t+1)
                break
        else:
            steps_to_converge.append(n_iters)
        final_errors.append(abs(pos - target))
    
    results[kappa] = {
        'mean_error': np.mean(final_errors),
        'std_error': np.std(final_errors),
        'mean_steps': np.mean(steps_to_converge),
        'std_steps': np.std(steps_to_converge)
    }

# 输出结果
print("\n【实验结果】")
print(f"{'κ':<6}{'最终误差(mean)':<15}{'误差波动(std)':<15}{'收敛步数':<12}")
for k in params:
    r = results[k]
    print(f"{k:<6}{r['mean_error']:<15.4f}{r['std_error']:<15.4f}{r['mean_steps']:<12.2f}")

# 可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.errorbar([str(k) for k in params], 
             [results[k]['mean_error'] for k in params],
             yerr=[results[k]['std_error'] for k in params],
             fmt='o-', capsize=5)
plt.xlabel('κ')
plt.ylabel('Final Error')
plt.title('SCAC-OPT: 最终误差 vs κ')

plt.subplot(1, 2, 2)
plt.errorbar([str(k) for k in params], 
             [results[k]['mean_steps'] for k in params],
             yerr=[results[k]['std_steps'] for k in params],
             fmt='o-', capsize=5, color='green')
plt.xlabel('κ')
plt.ylabel('Steps to Converge')
plt.title('SCAC-OPT: 收敛速度 vs κ')

plt.tight_layout()
plt.savefig('SCAC_OPT_results.png', dpi=150)
plt.show()

print("\n✅ 实验完成，结果已保存为 SCAC_OPT_results.png")
print("="*60)