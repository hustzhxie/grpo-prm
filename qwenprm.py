import numpy as np
import matplotlib.pyplot as plt

def shannon_entropy(p):
    """计算离散概率分布的香农熵"""
    p = np.array(p)
    # 确保概率和为1
    p = p / np.sum(p)
    # 避免log(0)的情况
    return -np.sum(p * np.log2(p + 1e-10))

# 绘制二项分布的熵函数
p_values = np.linspace(0.01, 0.99, 100)
entropy_values = [- (p * np.log2(p) + (1-p) * np.log2(1-p)) for p in p_values]

plt.figure(figsize=(10, 6))
plt.plot(p_values, entropy_values, 'b-', linewidth=2)
plt.title("二项分布的香农熵: H(p) = -[p·log₂p + (1-p)·log₂(1-p)]")
plt.xlabel("概率 p")
plt.ylabel("熵 H(p)")
plt.grid(True, alpha=0.3)
plt.savefig('function_plot.png') 