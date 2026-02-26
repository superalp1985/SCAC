import numpy as np
import matplotlib.pyplot as plt

# ==================== 实验1：语义鸿沟定理 ====================
print("="*60)
print("实验1：语义鸿沟定理 (Theorem 1: Semantic Gap)")
print("验证：连续意图 vs 离散表达之间的最小距离 > 0")
print("="*60)

# 你的原始数据（从你发的表格解析）
# 格式：[离散点数量, 5次重复实验的距离值]
raw_data = [
    [200, 0.000, 0.000, 0.000, 0.000, 0.000],
    [400, 0.000, 0.010, 0.010, 0.010, 0.010],
    [600, 0.000, 0.020, 0.020, 0.020, 0.020],
    [800, 0.000, 0.040, 0.040, 0.040, 0.040],
    [1000,0.000, 0.050, 0.050, 0.050, 0.050]
]

# 解析数据
n_points = [row[0] for row in raw_data]
distances = [row[1:] for row in raw_data]  # 5次重复实验
mean_distances = [np.mean(d) for d in distances]
std_distances = [np.std(d) for d in distances]

# 打印统计数据
print("\n【实验结果统计】")
print(f"{'离散点数量':<12}{'平均距离':<12}{'标准差':<12}")
for i, n in enumerate(n_points):
    print(f"{n:<12}{mean_distances[i]:<12.4f}{std_distances[i]:<12.4f}")

# 可视化
plt.figure(figsize=(10, 6))

# 绘制均值和误差条
plt.errorbar(n_points, mean_distances, yerr=std_distances, 
             fmt='o-', capsize=5, capthick=2, ecolor='red', 
             markersize=8, linewidth=2, label='Mean ± Std')

# 绘制每次实验的原始数据点（散点）
for i, n in enumerate(n_points):
    plt.scatter([n]*5, distances[i], alpha=0.5, s=30, color='blue', label='Individual runs' if i==0 else "")

# 添加理论参考线
x_theory = np.linspace(200, 1000, 100)
# 理论预期：距离应该随离散点增加而减小，但不为0
# 这里用 1/sqrt(n) 作为示意
y_theory = 0.5 / np.sqrt(x_theory)
plt.plot(x_theory, y_theory, 'g--', linewidth=1.5, label='Theoretical trend (1/√n)')

plt.xlabel('Number of Discrete Points', fontsize=12)
plt.ylabel('Minimum Distance', fontsize=12)
plt.title('Theorem 1: Semantic Gap', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('experiment1_semantic_gap.png', dpi=150)
plt.show()

print("\n✅ 实验完成！结果已保存为 experiment1_semantic_gap.png")
print("="*60)

# ==================== 简单分析 ====================
print("\n【初步分析】")
if mean_distances[-1] > 0:
    print("✅ 定理1验证：最小距离 > 0，语义鸿沟存在")
else:
    print("⚠️ 注意：最小距离 = 0，可能精度问题")

# 检查趋势
if mean_distances[0] > mean_distances[-1]:
    print("✅ 趋势正确：离散点越多，距离越小")
else:
    print("⚠️ 趋势异常：离散点增加但距离未减小")

print("="*60)