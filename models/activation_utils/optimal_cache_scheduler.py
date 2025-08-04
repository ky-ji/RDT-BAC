import numpy as np
import logging
import pickle
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

class OptimalCacheScheduler:
    """
    使用动态规划算法计算最优缓存更新步骤。
    
    基于激活值相似度矩阵，计算最优的缓存更新时间步。
    对于mse、l1、l2等距离度量，最小化总相似度；对于cosine等相似度度量，最大化总相似度。
    """
    
    def __init__(self, similarity_matrix: np.ndarray = None, metric: str = 'cosine'):
        """
        初始化最优缓存调度器。
        
        Args:
            similarity_matrix: 时间步之间的相似度矩阵，形状为[n_steps, n_steps]
            metric: 使用的相似度指标，默认为'cosine'，可选['mse', 'cosine', 'l1', 'l2']
        """
        self.similarity_matrix = similarity_matrix
        self.metric = metric
        self.optimal_steps = None

    def set_similarity_matrix(self, similarity_matrix: np.ndarray):
        """设置相似度矩阵"""
        self.similarity_matrix = similarity_matrix

    def compute_optimal_steps(self, num_caches: int) -> List[int]:
        """
        计算最优的缓存更新步骤。
        
        Args:
            num_caches: 允许的缓存更新次数
            
        Returns:
            最优缓存更新步骤列表，长度为num_caches
        """
        if self.similarity_matrix is None:
            logger.error("未设置相似度矩阵，无法计算最优步骤")
            return []
            
        n_steps = self.similarity_matrix.shape[0]
        
        # 计算代价矩阵：cost[a][b] 表示从步骤a到步骤b使用缓存的总代价
        cost = np.zeros((n_steps, n_steps))
        for a in range(n_steps):
            for b in range(a, n_steps):
                # 代价是缓存步骤a与所有步骤a到b的相似度之和
                cost[a, b] = sum(self.similarity_matrix[a, t] for t in range(a, b+1))
        
        # 根据指标类型确定是最小化还是最大化
        minimize = self.metric in ['mse', 'l1', 'l2', 'wasserstein']
        
        # 动态规划表：dp[k][i] 表示使用k个缓存到达步骤i的最优代价
        if minimize:
            dp = np.full((num_caches+1, n_steps), float('inf'))
        else:
            dp = np.zeros((num_caches+1, n_steps))
        
        # 路径记录表：path[k][i] 记录到达dp[k][i]的前一个缓存步骤
        path = np.zeros((num_caches+1, n_steps), dtype=int)
        
        # 初始化：使用1个缓存（步骤0）
        for i in range(n_steps):
            dp[1, i] = cost[0, i]
        
        # 填充动态规划表
        for k in range(2, num_caches+1):
            for i in range(k-1, n_steps):  # 确保至少有k-1步可以放置前k-1个缓存
                best_val = float('inf') if minimize else -float('inf')
                best_j = 0
                
                for j in range(k-2, i):  # j是第k-1个缓存的位置
                    val = dp[k-1, j] + cost[j+1, i]
                    
                    if (minimize and val < best_val) or (not minimize and val > best_val):
                        best_val = val
                        best_j = j
                
                dp[k, i] = best_val
                path[k, i] = best_j
        
        # 回溯最优路径
        optimal_steps = []
        k, i = num_caches, n_steps-1
        
        while k > 0:
            if k == 1:
                optimal_steps.append(0)  # 第一个缓存总是在步骤0
                break
            else:
                j = path[k, i]
                optimal_steps.append(j+1)  # j+1是当前缓存步骤
                i, k = j, k-1
        
        optimal_steps = sorted(optimal_steps)
        self.optimal_steps = optimal_steps
        return optimal_steps

    def save_optimal_steps(self, save_path: str):
        """保存最优步骤到文件"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.optimal_steps, f)
        logger.info(f"最优步骤已保存到 {save_path}")

    @staticmethod
    def load_optimal_steps(load_path: str) -> List[int]:
        """从文件加载最优步骤"""
        with open(load_path, 'rb') as f:
            steps = pickle.load(f)
        return steps

def compute_optimal_cache_steps(similarity_matrix: np.ndarray, num_caches: int, metric: str = 'cosine') -> List[int]:
    """
    便捷函数：直接计算最优缓存步骤
    
    Args:
        similarity_matrix: 相似度矩阵
        num_caches: 缓存数量
        metric: 相似度指标
        
    Returns:
        最优步骤列表
    """
    scheduler = OptimalCacheScheduler(similarity_matrix, metric)
    return scheduler.compute_optimal_steps(num_caches) 