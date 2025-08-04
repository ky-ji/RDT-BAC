#!/usr/bin/env python3
"""
RDT缓存加速完整Pipeline脚本
整合激活收集、最优步数计算和评估的完整流程

使用方法:
python scripts/rdt_cache_pipeline.py \
  --checkpoint /path/to/model.pt \
  --task_name PickCube-v1 \
  --output_base_dir ./rdt_cache_results \
  --device cuda:0 \
  --modes threshold optimal \
  --run_evaluation

功能:
1. 收集原始激活
2. 计算最优缓存步数
3. 运行threshold和optimal模式的评估
4. 生成对比报告
"""

import os
import sys
import logging
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.activation_utils.collect_activations_rdt import collect_rdt_activations
from models.activation_utils.get_optimal_cache_update_steps_rdt import get_optimal_cache_update_steps_rdt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rdt_cache_pipeline")

def parse_args():
    parser = argparse.ArgumentParser(description='RDT缓存加速完整Pipeline')
    
    # 基础参数
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='预训练模型路径')
    parser.add_argument('--task_name', type=str, default='PickCube-v1',
                        help='任务名称')
    parser.add_argument('--output_base_dir', type=str, required=True,
                        help='输出基础目录')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    
    # Pipeline控制
    parser.add_argument('--modes', nargs='+', 
                        choices=['threshold', 'optimal'],
                        default=['threshold', 'optimal'],
                        help='要测试的缓存模式')
    parser.add_argument('--skip_activation_collection', action='store_true',
                        help='跳过激活收集（如果已存在）')
    parser.add_argument('--skip_optimal_computation', action='store_true',
                        help='跳过最优步数计算（如果已存在）')
    parser.add_argument('--run_evaluation', action='store_true',
                        help='运行评估测试')
    parser.add_argument('--benchmark_only', action='store_true',
                        help='只运行性能基准测试，不运行环境评估')
    
    # 缓存参数
    parser.add_argument('--cache_thresholds', nargs='+', type=int,
                        default=[3, 5, 8, 10],
                        help='threshold模式的缓存阈值列表')
    parser.add_argument('--metrics', nargs='+', 
                        default=['cosine', 'mse'],
                        help='最优步数计算的相似度指标')
    parser.add_argument('--num_caches_list', nargs='+', type=int,
                        default=[5, 10, 20, 30],
                        help='最优模式的缓存数量列表')

    
    # 评估参数
    parser.add_argument('--num_trials_benchmark', type=int, default=10,
                        help='性能基准测试的试验次数')
    parser.add_argument('--num_traj_env', type=int, default=10,
                        help='环境评估的轨迹数量')
    
    return parser.parse_args()

def run_activation_collection(args):
    """步骤1: 收集原始激活"""
    logger.info("=== 步骤1: 收集原始激活 ===")
    
    activations_path = collect_rdt_activations(
        checkpoint=args.checkpoint,
        output_base_dir=os.path.join(args.output_base_dir, 'activations'),
        task_name=args.task_name,
        device=args.device,
        force_recompute=not args.skip_activation_collection,
        cache_mode='original',
        random_seed=args.random_seed
    )
    
    logger.info(f"激活收集完成: {activations_path}")
    return activations_path

def run_optimal_computation(args, activations_path):
    """步骤2: 计算最优步数"""
    logger.info("=== 步骤2: 计算最优步数 ===")
    
    if 'optimal' not in args.modes:
        logger.info("跳过最优步数计算（未选择optimal模式）")
        return None
    
    optimal_steps_dir = os.path.join(args.output_base_dir, 'optimal_steps')
    
    try:
        optimal_steps = get_optimal_cache_update_steps_rdt(
            checkpoint=args.checkpoint,
            output_dir=optimal_steps_dir,
            task_name=args.task_name,
            device=args.device,
            force_recompute=not args.skip_optimal_computation,
            metrics=args.metrics,
            num_caches_list=args.num_caches_list,
            random_seed=args.random_seed
        )
        
        logger.info(f"最优步数计算完成: {optimal_steps_dir}")
        return os.path.join(optimal_steps_dir, 'optimal_steps')
        
    except Exception as e:
        logger.error(f"最优步数计算失败: {e}")
        return None

def run_single_evaluation(args, mode_config):
    """运行单个配置的评估"""
    mode = mode_config['mode']
    config = mode_config['config']
    
    logger.info(f"评估模式: {mode}, 配置: {config}")
    
    # 构建输出目录
    config_name = f"{mode}"
    if mode == 'threshold':
        config_name += f"_th{config['cache_threshold']}"
    elif mode == 'optimal':
        config_name += f"_{config['metric']}_{config['num_caches']}"

    
    output_dir = os.path.join(args.output_base_dir, 'evaluations', config_name)
    
    # 构建评估命令
    cmd_args = [
        '--pretrained_path', args.checkpoint,
        '--env-id', args.task_name,
        '--output_dir', output_dir,
        '--cache_mode', mode,
        '--random_seed', str(args.random_seed),
        '--num-traj', str(args.num_traj_env)
    ]
    
    if args.benchmark_only:
        cmd_args.append('--benchmark_only')
    
    # 添加模式特定参数
    if mode == 'threshold':
        cmd_args.extend(['--cache_threshold', str(config['cache_threshold'])])
    elif mode == 'optimal':
        cmd_args.extend([
            '--optimal_steps_dir', config['optimal_steps_dir'],
            '--num_caches', str(config['num_caches']),
            '--cache_metric', config['metric']
        ])

    
    # 运行评估
    try:
        import subprocess
        eval_script = project_root / 'eval_sim' / 'eval_rdt_maniskill_cache.py'
        cmd = ['python', str(eval_script)] + cmd_args
        
        logger.info(f"运行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            logger.info(f"评估成功: {config_name}")
            
            # 加载结果
            results_path = os.path.join(output_dir, 'evaluation_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                return {
                    'config_name': config_name,
                    'mode': mode,
                    'config': config,
                    'results': results
                }
        else:
            logger.error(f"评估失败: {config_name}")
            logger.error(f"错误输出: {result.stderr}")
            
    except Exception as e:
        logger.error(f"运行评估时出错: {e}")
    
    return None

def run_evaluations(args, optimal_steps_base_dir):
    """步骤3: 运行评估"""
    logger.info("=== 步骤3: 运行评估 ===")
    
    if not args.run_evaluation:
        logger.info("跳过评估（未指定--run_evaluation）")
        return []
    
    # 构建评估配置列表
    eval_configs = []
    
    # 原始模式
    if 'original' in args.modes:
        eval_configs.append({
            'mode': 'original',
            'config': {}
        })
    
    # threshold模式
    if 'threshold' in args.modes:
        for threshold in args.cache_thresholds:
            eval_configs.append({
                'mode': 'threshold',
                'config': {'cache_threshold': threshold}
            })
    
    # optimal模式
    if 'optimal' in args.modes and optimal_steps_base_dir:
        for metric in args.metrics:
            for num_caches in args.num_caches_list:
                optimal_steps_dir = os.path.join(optimal_steps_base_dir, 'optimal_steps')
                eval_configs.append({
                    'mode': 'optimal',
                    'config': {
                        'optimal_steps_dir': optimal_steps_dir,
                        'metric': metric,
                        'num_caches': num_caches
                    }
                })
    

    
    # 运行所有评估
    all_results = []
    for i, config in enumerate(eval_configs):
        logger.info(f"运行评估 {i+1}/{len(eval_configs)}")
        result = run_single_evaluation(args, config)
        if result:
            all_results.append(result)
    
    return all_results

def generate_report(args, all_results):
    """生成对比报告"""
    logger.info("=== 生成对比报告 ===")
    
    if not all_results:
        logger.warning("没有评估结果，跳过报告生成")
        return
    
    report_dir = Path(args.output_base_dir) / 'reports'
    report_dir.mkdir(exist_ok=True)
    
    # 生成JSON报告
    json_report = {
        'task_name': args.task_name,
        'checkpoint': args.checkpoint,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': all_results
    }
    
    json_path = report_dir / 'comparison_report.json'
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    # 生成文本报告
    txt_path = report_dir / 'comparison_report.txt'
    with open(txt_path, 'w') as f:
        f.write("RDT缓存加速对比报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"任务: {args.task_name}\n")
        f.write(f"模型: {args.checkpoint}\n")
        f.write(f"时间: {json_report['timestamp']}\n\n")
        
        # 性能对比表格
        f.write("性能对比\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'配置':<20} {'平均时间(s)':<12} {'频率(FPS)':<12} {'成功率(%)':<12}\n")
        f.write("-" * 60 + "\n")
        
        for result in all_results:
            config_name = result['config_name']
            benchmark = result['results'].get('benchmark', {})
            env_results = result['results'].get('environment', {})
            
            avg_time = benchmark.get('avg_time', 0)
            frequency = benchmark.get('frequency', 0)
            success_rate = env_results.get('success_rate', 0) if env_results else 0
            
            f.write(f"{config_name:<20} {avg_time:<12.4f} {frequency:<12.2f} {success_rate:<12.1f}\n")
        
        f.write("\n详细配置\n")
        f.write("-" * 30 + "\n")
        for result in all_results:
            f.write(f"\n{result['config_name']}:\n")
            f.write(f"  模式: {result['mode']}\n")
            for key, value in result['config'].items():
                f.write(f"  {key}: {value}\n")
    
    logger.info(f"报告已生成:")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  TXT: {txt_path}")

def main():
    args = parse_args()
    
    logger.info("=== RDT缓存加速Pipeline开始 ===")
    logger.info(f"任务: {args.task_name}")
    logger.info(f"模型: {args.checkpoint}")
    logger.info(f"输出目录: {args.output_base_dir}")
    logger.info(f"模式: {args.modes}")
    
    # 创建输出目录
    Path(args.output_base_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 步骤1: 收集激活
        activations_path = run_activation_collection(args)
        
        # 步骤2: 计算最优步数
        optimal_steps_dir = run_optimal_computation(args, activations_path)
        
        # 步骤3: 运行评估
        all_results = run_evaluations(args, optimal_steps_dir)
        
        # 步骤4: 生成报告
        generate_report(args, all_results)
        
        logger.info("=== Pipeline完成 ===")
        
    except Exception as e:
        logger.error(f"Pipeline执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 