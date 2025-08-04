#!/usr/bin/env python3

"""
RDT模型最优缓存步数计算脚本
参考Fast_diffusion_policy实现，统一路径结构和BU算法支持

使用方法:

    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
    
    
python -m models.activation_utils.get_optimal_cache_update_steps_rdt \
  --checkpoint /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt \
  --output_dir assets/PegInsertionSide-v1 \
  --device cpu \
  --metrics cosine \
  --num_caches 20 \
  --force_recompute
  
  
python -m models.activation_utils.get_optimal_cache_update_steps_rdt \
  --checkpoint /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt \
  --output_dir assets/PickCube-v1 \
  --device cpu \
  --metrics cosine \
  --num_caches 20 \
  --force_recompute
  

  
python -m models.activation_utils.get_optimal_cache_update_steps_rdt \
  --checkpoint /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt \
  --output_dir assets/StackCube-v1 \
  --device cpu \
  --metrics cosine \
  --num_caches 20 \
  --force_recompute
  
  
python -m models.activation_utils.get_optimal_cache_update_steps_rdt \
  --checkpoint /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt \
  --output_dir assets/PlugCharger-v1 \
  --device cpu \
  --metrics cosine \
  --num_caches 20 \
  --force_recompute
  
python -m models.activation_utils.get_optimal_cache_update_steps_rdt \
  --checkpoint /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt \
  --output_dir assets/PushCube-v1 \
  --device cpu \
  --metrics cosine \
  --num_caches 20 \
  --force_recompute
"""


import os
import sys
import logging
import pickle
import numpy as np
import click
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.activation_utils.collect_activations_rdt import collect_rdt_activations
from models.activation_utils.optimal_cache_scheduler import OptimalCacheScheduler, compute_optimal_cache_steps
from models.activation_utils.activation_analysis import (
    compute_similarity_matrix,
    load_activations,
    save_similarity_matrices,
    get_module_metrics_from_similarity_data
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_similarity_matrix(similarity_matrices_dir: str, module_name: str, metric: str) -> Optional[np.ndarray]:
    """
    从相似度矩阵文件中加载特定模块和指标的相似度矩阵
    
    Args:
        similarity_matrices_dir: 相似度矩阵目录
        module_name: 模块名称
        metric: 指标名称
        
    Returns:
        相似度矩阵，如果不存在则返回None
    """
    similarity_file = Path(similarity_matrices_dir) / 'similarity_matrices.pkl'
    
    if not similarity_file.exists():
        logger.error(f"相似度文件不存在: {similarity_file}")
        return None
    
    try:
        with open(similarity_file, 'rb') as f:
            similarity_data = pickle.load(f)
        
        if module_name not in similarity_data:
            logger.error(f"模块 {module_name} 不存在于相似度数据中")
            return None
            
        module_data = similarity_data[module_name]
        
        if isinstance(module_data, dict):
            if metric not in module_data:
                logger.error(f"指标 {metric} 不存在于模块 {module_name} 的数据中")
                return None
            return module_data[metric]
        else:
            # 兼容旧格式
            if metric == 'cosine':
                return module_data
            else:
                logger.error(f"旧格式数据不支持指标 {metric}")
                return None
                
    except Exception as e:
        logger.error(f"加载相似度矩阵时出错: {str(e)}")
        return None


def get_optimal_cache_update_steps_rdt(
    checkpoint: str,
    output_dir: str,
    device: str = 'cuda:0',
    demo_idx: int = 0,
    force_recompute: bool = False,
    metrics: List[str] = None,
    num_caches_list: List[int] = None,
    task_name: str = None
) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    """
    简化版的RDT缓存分析工具，仅计算关键模块的最优缓存步骤
    参考Fast_diffusion_policy实现
    
    Args:
        checkpoint: 模型检查点路径
        output_dir: 输出目录
        device: 设备，默认为'cuda:0'
        demo_idx: 样本索引，默认为0
        force_recompute: 是否强制重新计算所有步骤，默认为False
        metrics: 要计算的相似度指标列表
        num_caches_list: 要计算的缓存数量列表
        task_name: 任务名称
        
    Returns:
        包含所有计算结果的字典，结构为 {metric: {module_name: {num_caches: optimal_steps}}}
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 从输出目录推断任务名
    if task_name is None:
        task_name = output_dir.name
    
    # 步骤1: 收集激活值
    activations_path = output_dir / 'original' / 'activations.pkl'
    if not activations_path.exists():
        logger.info(f"正在收集激活值...")
        # 提取任务名称
        output_base_dir = str(output_dir.parent)
        
      
        collect_rdt_activations(
            checkpoint=checkpoint, 
            output_base_dir=output_base_dir, 
            task_name=task_name,
            device=device, 
            demo_idx=demo_idx, 
            force_recompute=force_recompute
        )
        logger.info(f"激活值已保存到 {activations_path}")
        
    else:
        logger.info(f"找到现有激活值文件: {activations_path}")
    
    similarity_matrices_path = output_dir / 'original' / 'similarity_matrices.pkl'
    
    if not similarity_matrices_path.exists() or force_recompute:
        logger.info(f"正在计算相似度矩阵...")
        # 创建相似度矩阵目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载激活值
        activations_dict = load_activations(str(activations_path))
        
        # 仅为RDT模型关键模块计算相似度矩阵
        results = {}
        # 修改正则表达式以匹配实际收集的关键RDT组件
        pattern = re.compile(r'blocks\.\d+\.(attn|cross_attn|ffn)$')
        
        for module_name, module_activations in activations_dict.items():
            # 只处理RDT模型的关键模块
            if not pattern.match(module_name):
                logger.info(f"跳过模块: {module_name}，不是目标模块")
                continue
                
            logger.info(f"处理模块: {module_name}")
            
            try:
                # 计算相似度矩阵
                sim_matrices = compute_similarity_matrix(module_activations, 'all')
                results[module_name] = sim_matrices
            except Exception as e:
                logger.error(f"计算模块 {module_name} 的相似度矩阵时出错: {str(e)}")
                continue
        
        # 保存相似度矩阵
        save_similarity_matrices(results, similarity_matrices_path)
        logger.info(f"相似度矩阵已保存到 {similarity_matrices_path}")
    else:
        logger.info(f"找到现有相似度矩阵文件: {similarity_matrices_path}")
    
    # 获取相似度数据中的所有可用模块和指标
    module_metrics = get_module_metrics_from_similarity_data(str(output_dir / 'original'))
    print(module_metrics.keys())
    # 筛选出RDT模型的关键模块
    pattern = re.compile(r'blocks\.\d+\.(attn|cross_attn|ffn)$')
    module_names = [m for m in module_metrics.keys() if pattern.match(m)]
    
    if not module_names:
        logger.warning("没有找到RDT模型的关键模块，请检查激活值收集是否正确")
        return {}
        
    logger.info(f"将为以下模块计算最优缓存步骤: {module_names}")
    
    # 步骤3: 计算最优缓存步骤
    results = {}
    
    for metric in metrics:
        if metric not in results:
            results[metric] = {}
            
        for module_name in module_names:
            # 确保该模块支持所请求的指标
            if module_name in module_metrics and metric in module_metrics[module_name]:
                if module_name not in results[metric]:
                    results[metric][module_name] = {}
                    
                for num_caches in num_caches_list:
                    # 构建缓存步骤文件路径
                    cache_filename = f"optimal_steps_{module_name}_{num_caches}_{metric}.pkl"
                    cache_path = output_dir / 'optimal_steps'/ metric / module_name / cache_filename
                    
                    if not cache_path.exists() or force_recompute:
                        logger.info(f"计算最优缓存步骤: 模块={module_name}, 指标={metric}, 缓存数量={num_caches}")
                        try:
                            # 加载相似度矩阵
                            similarity_matrix = load_similarity_matrix(
                                similarity_matrices_dir=str(output_dir/'original'),
                                module_name=module_name,
                                metric=metric
                            )
                            
                            if similarity_matrix is None:
                                logger.error(f"无法加载模块 {module_name} 指标 {metric} 的相似度矩阵")
                                continue
                            
                            # 计算最优步骤
                            optimal_steps = compute_optimal_cache_steps(
                                similarity_matrix=similarity_matrix,
                                num_caches=num_caches,
                                metric=metric
                            )
                            
                            results[metric][module_name][num_caches] = optimal_steps
                            
                            # 保存结果到文件
                            cache_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(cache_path, 'wb') as f:
                                pickle.dump(optimal_steps, f)
                            
                            logger.info(f"最优缓存步骤已计算并保存到 {cache_path}")
                        except Exception as e:
                            logger.error(f"计算最优缓存步骤时出错: {str(e)}")
                            continue
                    else:
                        logger.info(f"找到现有最优缓存步骤文件: {cache_path}")
                        # 加载现有的最优步骤
                        try:
                            optimal_steps = OptimalCacheScheduler.load_optimal_steps(str(cache_path))
                            if optimal_steps:
                                results[metric][module_name][num_caches] = optimal_steps
                            else:
                                logger.warning(f"无法加载最优步骤文件 {cache_path}")
                        except Exception as e:
                            logger.error(f"加载最优步骤文件时出错: {str(e)}")
            else:
                logger.warning(f"模块 {module_name} 不支持指标 {metric}")
    
    return results


@click.command()
@click.option('-c', '--checkpoint', required=True, help='模型检查点路径')
@click.option('-o', '--output_dir', required=True, help='输出目录')
@click.option('-d', '--device', default='cuda:0', help='设备')
@click.option('--demo_idx', default=0, type=int, help='样本索引')
@click.option('--force_recompute', is_flag=True, help='强制重新计算所有步骤')
@click.option('--metrics', default='cosine,mse,l1', help='要计算的相似度指标，多个指标用逗号分隔，例如：cosine,mse,l1')
@click.option('--num_caches', default='5,8,10,20', help='要计算的缓存数量，多个数量用逗号分隔，例如：5,8,10,20')
@click.option('--task_name', default=None, help='任务名称，如果不提供将从输出目录推断')
def main(checkpoint, output_dir, device, demo_idx, force_recompute, metrics, num_caches, task_name):
    """简化版RDT缓存分析工具，仅计算关键模块的最优缓存步骤"""
    # 解析metrics字符串为列表
    metrics_list = metrics.split(',') if metrics else None
    
    # 解析num_caches字符串为整数列表
    num_caches_list = [int(n.strip()) for n in num_caches.split(',')] if num_caches else None
    
    # 执行整合分析
    results = get_optimal_cache_update_steps_rdt(
        checkpoint=checkpoint,
        output_dir=output_dir,
        device=device,
        demo_idx=demo_idx,
        force_recompute=force_recompute,
        metrics=metrics_list,
        num_caches_list=num_caches_list,
        task_name=task_name
    )
    
    # 输出分析结果摘要
    logger.info("分析完成。结果摘要:")
    if not results:
        logger.warning("没有生成任何结果")
        return
        
    for metric, module_dict in results.items():
        logger.info(f"指标: {metric}")
        for module_name, cache_dict in module_dict.items():
            logger.info(f"  模块: {module_name}")
            for num_caches, optimal_steps in cache_dict.items():
                logger.info(f"    缓存数量: {num_caches}, 步骤数: {len(optimal_steps)}")
                if len(optimal_steps) <= 10:
                    logger.info(f"      步骤: {optimal_steps}")
                else:
                    logger.info(f"      前10个步骤: {optimal_steps[:10]}...")
                
                # 显示可查看的文件路径
                cache_filename = f"optimal_steps_{module_name}_{num_caches}_{metric}"
                txt_path = os.path.join(output_dir, 'optimal_steps', metric, module_name, f"{cache_filename}.txt")
                json_path = os.path.join(output_dir, 'optimal_steps', metric, module_name, f"{cache_filename}.json")
                
                if os.path.exists(txt_path):
                    logger.info(f"      查看TXT文件: {txt_path}")
                if os.path.exists(json_path):
                    logger.info(f"      查看JSON文件: {json_path}")


if __name__ == '__main__':
    main() 