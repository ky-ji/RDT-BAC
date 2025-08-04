#!/usr/bin/env python3

"""
RDT模型激活分析工具
参考Fast_diffusion_policy实现，提供统一的激活分析功能
"""

import os
import sys
import pickle
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

def load_activations(activations_path: str) -> Dict[str, List[torch.Tensor]]:
    """
    加载激活值文件
    
    Args:
        activations_path: 激活值文件路径
        
    Returns:
        激活值字典 {module_name: [tensor_t0, tensor_t1, ...]}
    """
    with open(activations_path, 'rb') as f:
        activations = pickle.load(f)
    logger.info(f"从 {activations_path} 加载了 {len(activations)} 个模块的激活值")
    return activations

def compute_similarity_matrix(activations: List[torch.Tensor], metric: str = 'cosine') -> Dict[str, np.ndarray]:
    """
    计算激活序列的相似度矩阵
    
    Args:
        activations: 激活列表，每个元素是一个时间步的激活
        metric: 相似度指标 ('cosine', 'mse', 'l1')
    
    Returns:
        包含多个指标的相似度矩阵字典
    """
    n_steps = len(activations)
    results = {}
    
    # 支持的指标列表
    metrics = ['cosine', 'mse', 'l1'] if metric == 'all' else [metric]
    
    for current_metric in metrics:
        similarity_matrix = np.zeros((n_steps, n_steps))
        
        for i in range(n_steps):
            for j in range(n_steps):
                if i == j:
                    similarity_matrix[i, j] = 1.0 if current_metric == 'cosine' else 0.0
                    continue
                
                # 获取激活张量
                act_i = activations[i]
                act_j = activations[j]
                
                # 确保张量在CPU上
                if isinstance(act_i, torch.Tensor):
                    act_i = act_i.cpu().numpy()
                if isinstance(act_j, torch.Tensor):
                    act_j = act_j.cpu().numpy()
                
                # 展平张量
                flat_i = act_i.flatten()
                flat_j = act_j.flatten()
                
                # 计算相似度
                if current_metric == 'cosine':
                    # 余弦相似度
                    dot_product = np.dot(flat_i, flat_j)
                    norm_i = np.linalg.norm(flat_i)
                    norm_j = np.linalg.norm(flat_j)
                    if norm_i == 0 or norm_j == 0:
                        similarity = 0.0
                    else:
                        similarity = dot_product / (norm_i * norm_j)
                elif current_metric == 'mse':
                    # 均方误差（转换为相似度，越小越相似）
                    mse = np.mean((flat_i - flat_j) ** 2)
                    similarity = -mse  # 负号使得越小的MSE对应越高的相似度
                elif current_metric == 'l1':
                    # L1距离（转换为相似度）
                    l1_dist = np.mean(np.abs(flat_i - flat_j))
                    similarity = -l1_dist  # 负号使得越小的L1距离对应越高的相似度
                else:
                    raise ValueError(f"不支持的相似度指标: {current_metric}")
                
                similarity_matrix[i, j] = similarity
        
        results[current_metric] = similarity_matrix
    
    return results

def save_similarity_matrices(similarity_data: Dict[str, Dict[str, np.ndarray]], output_path: str):
    """
    保存相似度矩阵
    
    Args:
        similarity_data: 相似度数据 {module_name: {metric: matrix}}
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(similarity_data, f)
    logger.info(f"相似度矩阵已保存到 {output_path}")

def get_module_metrics_from_similarity_data(similarity_dir: str) -> Dict[str, List[str]]:
    """
    从相似度数据目录获取模块和指标信息
    
    Args:
        similarity_dir: 相似度数据目录
        
    Returns:
        模块名称到支持指标列表的字典
    """
    similarity_file = Path(similarity_dir) / 'similarity_matrices.pkl'
    
    if not similarity_file.exists():
        logger.warning(f"相似度文件不存在: {similarity_file}")
        return {}
    
    with open(similarity_file, 'rb') as f:
        similarity_data = pickle.load(f)
    
    module_metrics = {}
    for module_name, module_data in similarity_data.items():
        if isinstance(module_data, dict):
            module_metrics[module_name] = list(module_data.keys())
        else:
            # 兼容旧格式（直接是矩阵）
            module_metrics[module_name] = ['cosine']
    
    return module_metrics

def compute_activation_statistics(activations: Dict[str, List[torch.Tensor]]) -> Dict[str, Dict[str, Any]]:
    """
    计算激活值统计信息
    
    Args:
        activations: 激活值字典
        
    Returns:
        统计信息字典
    """
    stats = {}
    
    for module_name, module_activations in activations.items():
        module_stats = {
            'num_timesteps': len(module_activations),
            'shapes': [],
            'mean_values': [],
            'std_values': [],
            'min_values': [],
            'max_values': []
        }
        
        for act in module_activations:
            if isinstance(act, torch.Tensor):
                act = act.cpu().numpy()
            
            module_stats['shapes'].append(act.shape)
            module_stats['mean_values'].append(float(np.mean(act)))
            module_stats['std_values'].append(float(np.std(act)))
            module_stats['min_values'].append(float(np.min(act)))
            module_stats['max_values'].append(float(np.max(act)))
        
        stats[module_name] = module_stats
    
    return stats

def analyze_activation_similarity_evolution(activations: List[torch.Tensor], 
                                          metric: str = 'cosine') -> Dict[str, Any]:
    """
    分析激活值相似度随时间的演变
    
    Args:
        activations: 激活值列表
        metric: 相似度指标
        
    Returns:
        相似度演变分析结果
    """
    n_steps = len(activations)
    if n_steps < 2:
        return {'error': 'Need at least 2 timesteps for similarity analysis'}
    
    # 计算相邻时间步的相似度
    adjacent_similarities = []
    for i in range(n_steps - 1):
        sim_matrices = compute_similarity_matrix([activations[i], activations[i+1]], metric)
        adjacent_similarities.append(sim_matrices[metric][0, 1])
    
    # 计算与第一个时间步的相似度
    initial_similarities = []
    for i in range(1, n_steps):
        sim_matrices = compute_similarity_matrix([activations[0], activations[i]], metric)
        initial_similarities.append(sim_matrices[metric][0, 1])
    
    return {
        'adjacent_similarities': adjacent_similarities,
        'initial_similarities': initial_similarities,
        'mean_adjacent_similarity': np.mean(adjacent_similarities),
        'std_adjacent_similarity': np.std(adjacent_similarities),
        'mean_initial_similarity': np.mean(initial_similarities),
        'std_initial_similarity': np.std(initial_similarities)
    } 