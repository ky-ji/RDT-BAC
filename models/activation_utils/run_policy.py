#!/usr/bin/env python3

"""
RDT模型统一策略运行接口
参考Fast_diffusion_policy的run_policy.py实现

提供统一的策略运行接口，支持不同的缓存模式
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from models.activation_utils.run_policy_rdt import run_rdt_policy
from models.acceleration.rdt_cache_wrapper import RDTCacheAccelerator

logger = logging.getLogger(__name__)

def run_policy(
    checkpoint: str,
    output_dir: str,
    device: str = 'cuda:0',
    demo_idx: int = 0,
    cache_mode: str = 'original',
    cache_threshold: int = 5,
    optimal_steps_dir: str = None,
    num_caches: int = 30,
    metric: str = 'cosine',
    num_bu_blocks: int = 3,
    edit_steps: List[int] = None,
    interpolation_ratio: float = 1.0,
    reference_activations_path: str = None,
    random_seed: int = 11
) -> Tuple[Any, Dict[str, torch.Tensor]]:
    """
    统一的策略运行接口，支持不同的缓存模式
    
    Args:
        checkpoint: 模型检查点路径
        output_dir: 输出目录
        device: 运行设备
        demo_idx: 演示索引
        cache_mode: 缓存模式
        cache_threshold: 缓存阈值
        optimal_steps_dir: 最优步骤目录
        num_caches: 缓存数量
        metric: 相似度指标
        num_bu_blocks: BU算法块数量
        edit_steps: 自定义步骤
        interpolation_ratio: 插值比例
        reference_activations_path: 参考激活路径
        random_seed: 随机种子
        
    Returns:
        (policy, obs_dict): 策略对象和观测字典
    """
    # 首先运行原始RDT策略以获取策略和输入数据
    policy, obs_dict = run_rdt_policy(
        checkpoint=checkpoint,
        output_dir=output_dir,
        device=device,
        demo_idx=demo_idx,
        task_name='PickCube-v1',  # 可以根据需要调整
        cache_mode='original',  # 先以原始模式运行
        cache_threshold=cache_threshold,
        return_obs_action=False,
        random_seed=random_seed
    )
    
    # 如果需要应用缓存，则应用相应的缓存模式
    if cache_mode != 'original':
        logger.info(f"应用缓存模式: {cache_mode}")
        
        # 应用缓存加速
        policy = RDTCacheAccelerator.apply_cache(
            policy=policy.policy,  # run_rdt_policy返回的是包装后的policy
            cache_threshold=cache_threshold,
            optimal_steps_dir=optimal_steps_dir,
            num_caches=num_caches,
            metric=metric,
            cache_mode=cache_mode,
            interpolation_ratio=interpolation_ratio,
            reference_activations_path=reference_activations_path,
            edit_steps=edit_steps,
            num_bu_blocks=num_bu_blocks
        )
    
    return policy, obs_dict 