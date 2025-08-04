#!/usr/bin/env python3

"""
RDT模型BU算法块选择工具
计算各个模块的L1范数误差，并选择误差最大的N个块应用BU算法。

参考Fast_diffusion_policy的bu_block_selection.py实现

使用示例:
# 计算误差并选择前5个块应用BU算法
python -m models.activation_utils.bu_block_selection -o assets -t PegInsertionSide-v1 --cache_mode original --num_blocks 3

python -m models.activation_utils.bu_block_selection -o assets -t PickCube-v1 --cache_mode original --num_blocks 3

python -m models.activation_utils.bu_block_selection -o assets -t StackCube-v1 --cache_mode original --num_blocks 10

python -m models.activation_utils.bu_block_selection -o assets -t PlugCharger-v1 --cache_mode original --num_blocks 3

python -m models.activation_utils.bu_block_selection -o assets -t PushCube-v1 --cache_mode original --num_blocks 3
"""

import sys
import os
import torch
import numpy as np
import pickle
from pathlib import Path
import logging
import click
import re
from tqdm import tqdm
from collections import defaultdict

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 导入工具函数
from models.activation_utils.activation_analysis import load_activations
# 移除不存在的函数导入
# from models.activation_utils.collect_activations_rdt import get_activations_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_block_l1_errors(original_activations, cached_activations=None):
    """
    计算每个模块的L1范数误差。
    
    Args:
        original_activations: 原始模型的激活值
        cached_activations: 缓存模型的激活值，如果为None则使用自身时间步之间的误差
        
    Returns:
        块名称到平均L1误差的字典
    """
    # 针对RDT模型结构，查找blocks.*的模块
    block_pattern = re.compile(r'blocks\.(\d+)\..*')
    
    # 收集块名称到激活值的映射
    block_activations = {}
    for module_name in original_activations.keys():
        match = block_pattern.match(module_name)
        if match:
            layer_num = int(match.group(1))
            
            # 对于RDT，我们关注不同类型的块
            if 'attn' in module_name:
                block_key = f"blocks.{layer_num}_attn_block"
            elif 'cross_attn' in module_name:
                block_key = f"blocks.{layer_num}_cross_attn_block"
            elif 'mlp' in module_name or 'ffn' in module_name:
                block_key = f"blocks.{layer_num}_ffn_block"
            else:
                # 其他类型的块
                block_key = f"blocks.{layer_num}_other_block"
            
            # 如果该块还没有激活值，初始化
            if block_key not in block_activations:
                block_activations[block_key] = original_activations[module_name]
    
    # 计算每个块的平均L1误差
    block_errors = {}
    
    # 计算每个块在不同时间步之间的误差
    for block_key, activations in block_activations.items():
        num_timesteps = len(activations)
        if num_timesteps <= 1:
            logger.warning(f"块 {block_key} 只有 {num_timesteps} 个时间步，跳过")
            continue
        
        total_error = 0.0
        total_comparisons = 0
        
        # 如果没有提供缓存激活值，则计算自身在不同时间步之间的平均误差
        if cached_activations is None:
            # 计算所有时间步对之间的L1误差
            for i in range(num_timesteps):
                for j in range(i+1, num_timesteps):
                    act_i = activations[i]
                    act_j = activations[j]
                    
                    # 转换为torch张量
                    if isinstance(act_i, np.ndarray):
                        act_i = torch.from_numpy(act_i)
                    elif isinstance(act_i, torch.Tensor):
                        act_i = act_i.cpu()
                    
                    if isinstance(act_j, np.ndarray):
                        act_j = torch.from_numpy(act_j)
                    elif isinstance(act_j, torch.Tensor):
                        act_j = act_j.cpu()
                    
                    # 跳过形状不匹配的激活值
                    if act_i.shape != act_j.shape:
                        continue
                    
                    # 计算L1误差
                    error = torch.mean(torch.abs(act_i - act_j)).item()
                    total_error += error
                    total_comparisons += 1
        else:
            # 如果提供了缓存激活值，计算原始激活值和缓存激活值之间的误差
            if block_key in cached_activations:
                cached_acts = cached_activations[block_key]
                min_timesteps = min(num_timesteps, len(cached_acts))
                
                for t in range(min_timesteps):
                    orig_act = activations[t]
                    cache_act = cached_acts[t]
                    
                    # 转换为torch张量
                    if isinstance(orig_act, np.ndarray):
                        orig_act = torch.from_numpy(orig_act)
                    elif isinstance(orig_act, torch.Tensor):
                        orig_act = orig_act.cpu()
                    
                    if isinstance(cache_act, np.ndarray):
                        cache_act = torch.from_numpy(cache_act)
                    elif isinstance(cache_act, torch.Tensor):
                        cache_act = cache_act.cpu()
                    
                    # 跳过形状不匹配的激活值
                    if orig_act.shape != cache_act.shape:
                        continue
                    
                    # 计算L1误差
                    error = torch.mean(torch.abs(orig_act - cache_act)).item()
                    total_error += error
                    total_comparisons += 1
        
        # 计算平均误差
        if total_comparisons > 0:
            block_errors[block_key] = total_error / total_comparisons
        else:
            logger.warning(f"块 {block_key} 没有有效的比较对，跳过")
    
    return block_errors

def select_top_error_blocks(block_errors, num_blocks=5):
    """
    选择误差最大的N个块。
    
    Args:
        block_errors: 块名称到误差的字典
        num_blocks: 要选择的块数量
        
    Returns:
        排序后的(块名称, 误差)元组列表
    """
    # 按误差从大到小排序
    sorted_errors = sorted(block_errors.items(), key=lambda x: x[1], reverse=True)
    
    # 返回前N个
    return sorted_errors[:num_blocks]

def save_block_errors(block_errors, output_path):
    """
    保存块误差数据。
    
    Args:
        block_errors: 块名称到误差的字典
        output_path: 保存路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(block_errors, f)
    logger.info(f"块误差数据已保存到: {output_path}")

def save_selected_blocks(selected_blocks, output_path):
    """
    保存选定的块。
    
    Args:
        selected_blocks: 选定的(块名称, 误差)元组列表
        output_path: 保存路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    selected_dict = {block: error for block, error in selected_blocks}
    with open(output_path, 'wb') as f:
        pickle.dump(selected_dict, f)
    logger.info(f"选定的块已保存到: {output_path}")

def analyze_block_errors(activations_path, output_dir, num_blocks=5):
    """
    分析块误差并选择误差最大的块。
    
    Args:
        activations_path: 激活值文件路径
        output_dir: 输出目录
        num_blocks: 要选择的块数量
        
    Returns:
        选定的块列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载激活值
    logger.info(f"加载激活值: {activations_path}")
    activations = load_activations(activations_path)
    
    # 计算块误差
    logger.info("计算块误差...")
    block_errors = compute_block_l1_errors(activations)
    
    # 保存全部块误差
    errors_path = output_dir / 'block_l1_errors.pkl'
    save_block_errors(block_errors, errors_path)
    
    # 选择误差最大的块
    logger.info(f"选择误差最大的 {num_blocks} 个块...")
    selected_blocks = select_top_error_blocks(block_errors, num_blocks)
    
    # 保存选定的块
    selected_path = output_dir / f'top_{num_blocks}_error_blocks.pkl'
    save_selected_blocks(selected_blocks, selected_path)
    
    # 输出选定的块
    logger.info(f"误差最大的 {num_blocks} 个块:")
    for block, error in selected_blocks:
        logger.info(f"  {block}: {error:.6f}")
    
    return selected_blocks

def get_analysis_output_dir(output_base_dir, task_name, cache_mode, **kwargs):
    """
    获取分析结果的输出目录路径。
    
    Args:
        output_base_dir: 基本输出目录
        task_name: 任务名称
        cache_mode: 缓存模式
        **kwargs: 根据缓存模式的其他参数
    
    Returns:
        分析结果的输出目录路径
    """
    # 构建激活值路径的目录
    activations_dir = Path(output_base_dir) / task_name / cache_mode
    # 构建分析结果输出目录
    return activations_dir / 'bu_block_selection'

@click.command()
@click.option('-o', '--output_base_dir', required=True, help='基本输出目录')
@click.option('-t', '--task_name', required=True, help='任务名称')
@click.option('--cache_mode', default='original', 
              type=click.Choice(['original', 'threshold', 'optimal']), 
              help='缓存模式')
@click.option('--cache_threshold', default=5, type=int, help='缓存阈值')
@click.option('--num_caches', default=30, type=int, help='缓存更新次数')
@click.option('--metric', default='cosine', help='相似度指标类型')
@click.option('--num_blocks', default=5, type=int, help='要选择的块数量')
@click.option('--force', is_flag=True, help='即使结果存在也强制重新计算')
def main(output_base_dir, task_name, cache_mode, cache_threshold, 
         num_caches, metric, num_blocks, force):
    """命令行工具，用于计算块误差并选择误差最大的块应用BU算法"""
    
    # 构建路径构造函数的参数字典
    kwargs = {
        'cache_threshold': cache_threshold,
        'num_caches': num_caches,
        'metric': metric
    }
    
    # 构建激活值文件路径
    activations_path = Path(output_base_dir) / task_name / cache_mode / 'activations.pkl'
    
    # 检查激活值文件是否存在
    if not activations_path.exists():
        logger.error(f"激活值文件不存在: {activations_path}")
        logger.error(f"请先运行collect_activations_rdt.py，使用cache_mode={cache_mode}")
        return
    
    # 获取分析结果输出目录
    analysis_output_dir = get_analysis_output_dir(output_base_dir, task_name, cache_mode, **kwargs)
    
    # 检查是否需要重新计算
    if analysis_output_dir.exists() and not force:
        selected_path = analysis_output_dir / f'top_{num_blocks}_error_blocks.pkl'
        if selected_path.exists():
            logger.info(f"分析结果已存在: {selected_path}")
            logger.info("使用--force强制重新计算")
            return
    
    # 创建输出目录
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 分析块误差
    logger.info(f"分析{cache_mode}模式的块误差")
    analyze_block_errors(str(activations_path), str(analysis_output_dir), num_blocks)
    
    logger.info(f"分析完成，结果保存到: {analysis_output_dir}")

if __name__ == '__main__':
    main() 