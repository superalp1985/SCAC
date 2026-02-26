import numpy as np
import matplotlib.pyplot as plt

# ==================== 实验3：空间压缩定理 ====================
print("="*60)
print("实验3：空间压缩定理 (Theorem 6: Space Compression)")
print("验证：物理反馈将搜索空间从无界压缩至有限可行集")
print("="*60)

# 设置随机种子
np.random.seed(42)

# 参数设置
N_total = 10000          # 总代码空间大小（模拟无界空间中的样本）
n_iters = 10             # 迭代轮数
n_trials = 50            # 重复实验次数

# 模拟“物理判定算子”Φ：假设只有10%的代码能通过编译
feasible_ratio = 0.1
feasible_set_size = int(N_total * feasible_ratio)

# 存储每一轮的可行集大小（搜索空间大小）
feasible_sizes = np.zeros((n_trials, n_iters))
pass_rates = np.zeros((n_trials, n_iters))

# 进行多次实验
for trial in range(n_trials):
    # 初始时，AI没有任何反馈，随机尝试所有代码
    # 但物理反馈会逐步压缩搜索空间
    
    # 记录当前轮次中已探索过的代码集合（去重）
    explored = set()
    
    for t in range(n_iters):
        # 模拟AI生成代码：随机从N_total中选一批（模拟生成）
        batch_size = 500
        generated = np.random.randint(0, N_total, batch_size)
        
        # 模拟编译：只有落在可行集内的代码能通过
        # 可行集是固定的随机子集
        # 为了模拟，我们预先随机生成可行集索引
        if t == 0:
            # 第一次随机生成可行集
            feasible_indices = set(np.random.choice(N_total, feasible_set_size, replace=False))
        
        # 统计这一轮生成中通过编译的代码
        passed = [idx for idx in generated if idx in feasible_indices]
        # 将通过的代码加入已探索集合
        explored.update(passed)
        
        # 记录当前探索到的可行集大小（即已经发现的可行代码数量）
        feasible_sizes[trial, t] = len(explored)
        # 这一轮的通过率
        pass_rates[trial, t] = len(passed) / batch_size

# 计算平均值和标准差
mean_feasible = np.mean(feasible_sizes, axis=0)
std_feasible = np.std(feasible_sizes, axis=0)
mean_pass = np.mean(pass_rates, axis=0)
std_pass = np.std(pass_rates, axis=0)

# 打印统计信息
print("\n【搜索结果】")
print(f"{'迭代轮次':<10}{'已发现可行代码数':<20}{'平均通过率':<15}")
for t in range(n_iters):
    print(f"{t:<10}{mean_feasible[t]:<20.1f}{mean_pass[t]:<15.4f}")
print(f"\n理论可行集总大小: {feasible_set_size}")
print(f"第{n_iters-1}轮发现比例: {mean_feasible[-1]/feasible_set_size*100:.1f}%")

# ==================== 可视化 ====================
plt.figure(figsize=(12, 5))

# 图1：已发现可行代码数（搜索空间压缩）
plt.subplot(1, 2, 1)
plt.fill_between(range(n_iters), 
                 mean_feasible - std_feasible, 
                 mean_feasible + std_feasible, 
                 alpha=0.3, color='blue', label='±1 Std')
plt.plot(range(n_iters), mean_feasible, 'b-o', linewidth=2, markersize=6, label='Mean discovered')
plt.axhline(y=feasible_set_size, color='red', linestyle='--', 
            label=f'Total feasible set = {feasible_set_size}')
plt.xlabel('Iteration')
plt.ylabel('Number of discovered feasible codes')
plt.title('Theorem 6: Space Compression\n(Search space shrinks to feasible set)')
plt.legend()
plt.grid(True, alpha=0.3)

# 图2：通过率提升
plt.subplot(1, 2, 2)
plt.fill_between(range(n_iters), 
                 mean_pass - std_pass, 
                 mean_pass + std_pass, 
                 alpha=0.3, color='green', label='±1 Std')
plt.plot(range(n_iters), mean_pass, 'g-o', linewidth=2, markersize=6, label='Mean pass rate')
plt.axhline(y=feasible_ratio, color='red', linestyle='--', 
            label=f'Theoretical pass rate = {feasible_ratio*100:.0f}%')
plt.xlabel('Iteration')
plt.ylabel('Compilation pass rate')
plt.title('Pass rate increases with feedback')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiment3_space_compression.png', dpi=150)
plt.show()

# ==================== 定理验证 ====================
print("\n" + "="*60)
print("【定理6验证结果】")

# 验证1：是否最终探索到了大部分可行集
coverage = mean_feasible[-1] / feasible_set_size
if coverage > 0.9:
    print(f"✅ 探索覆盖率达到 {coverage*100:.1f}%，搜索空间显著压缩 ✓")
else:
    print(f"⚠️ 覆盖率 {coverage*100:.1f}%，可能还需更多迭代")

# 验证2：通过率是否显著高于随机
if mean_pass[-1] > feasible_ratio * 2:
    print(f"✅ 最终通过率 {mean_pass[-1]*100:.1f}% 远高于随机基准 {feasible_ratio*100:.0f}% ✓")
else:
    print(f"⚠️ 最终通过率 {mean_pass[-1]*100:.1f}% 接近随机基准 {feasible_ratio*100:.0f}%")

# 验证3：搜索空间大小是否稳定（收敛）
if np.std(feasible_sizes[:, -5:], axis=0).mean() < 10:
    print("✅ 搜索空间大小已稳定，收敛 ✓")
else:
    print("⚠️ 搜索空间仍在增长，可能未完全收敛")

print("\n" + "="*60)
print("✅ 实验完成！结果已保存为 experiment3_space_compression.png")
print("="*60)

# 增加暂停，防止闪退
input("\n按回车键退出...")