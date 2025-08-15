#!/usr/bin/env python3
"""
RDT缓存加速评估脚本
参考Fast_diffusion_policy/scripts/eval_fast_diffusion_policy.py

重要提醒：
1. 运行前请确保激活正确的conda环境：conda activate rdt
2. BU算法现已正确实现，会自动从后层FFN Block获取steps进行传播

    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
    
使用方法:
# threshold缓存加速
python -m eval_sim.eval_rdt_maniskill_cache \
  --pretrained_path /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt \
  -e PegInsertionSide-v1 \
  --cache_mode threshold \
  --cache_threshold 20 \
  --output_dir ./results/rdt_threshold
  
  
  python -m eval_sim.eval_rdt_maniskill_cache   \
  --pretrained_path /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt   \
  -e StackCube-v1 \
  --output_dir ./results/rdt_original

# optimal缓存加速 + BU算法

python -m eval_sim.eval_rdt_maniskill_cache \
  --pretrained_path /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt \
  -e PegInsertionSide-v1 \
  --cache_mode optimal \
  --optimal_steps_dir ./assets/PegInsertionSide-v1/optimal_steps \
  --output_dir ./results/rdt_optimal_bu \
  --num_caches 2 \
  --num_bu_blocks 0
  


python -m eval_sim.eval_rdt_maniskill_cache \
  --pretrained_path /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt \
  -e PickCube-v1 \
  --cache_mode optimal \
  --optimal_steps_dir ./assets/PickCube-v1/optimal_steps \
  --output_dir ./results/rdt_optimal_bu \
  --num_caches 2 \
  --num_bu_blocks 0

python -m eval_sim.eval_rdt_maniskill_cache \
  --pretrained_path /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt \
  -e StackCube-v1 \
  --cache_mode optimal \
  --optimal_steps_dir ./assets/StackCube-v1/optimal_steps \
  --output_dir ./results/rdt_optimal_bu \
  --num_caches 2 \
  --num_bu_blocks 0
  
  python -m eval_sim.eval_rdt_maniskill_cache \
  --pretrained_path /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt \
  -e PlugCharger-v1 \
  --cache_mode optimal \
  --optimal_steps_dir ./assets/PlugCharger-v1/optimal_steps \
  --output_dir ./results/rdt_optimal_bu \
  --num_caches 2 \
  --num_bu_blocks 3 
  
  python -m eval_sim.eval_rdt_maniskill_cache \
  --pretrained_path /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt \
  -e PushCube-v1 \
  --cache_mode optimal \
  --optimal_steps_dir ./assets/PushCube-v1/optimal_steps \
  --output_dir ./results/rdt_optimal_bu \
  --num_caches 1 \
  --num_bu_blocks 0
  
  
  
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Callable, List, Type
import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
import argparse
import yaml
import torch
from collections import deque
from PIL import Image
import cv2
import random
import tqdm
from copy import deepcopy

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.maniskill_model import create_model, RoboticDiffusionTransformerModel
from models.acceleration.rdt_cache_wrapper import RDTCacheAccelerator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eval_rdt_cache")

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='RDT缓存加速评估脚本')
    
    # 基础环境参数
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", 
                        help="Environment to run evaluation on")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb", 
                        help="Observation mode")
    parser.add_argument("-n", "--num-traj", type=int, default=25, 
                        help="Number of trajectories to test")
    parser.add_argument("--only-count-success", action="store_true", 
                        help="Only count successful trajectories")
    parser.add_argument("--reward-mode", type=str, default="dense")
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", 
                        help="Simulation backend")
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--shader", default="default", type=str, 
                        help="Rendering shader")
    parser.add_argument("--num-procs", type=int, default=1, 
                        help="Number of parallel processes")
    
    # 模型参数
    parser.add_argument("--pretrained_path", type=str, default='/home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt', 
                        help="Path to the pretrained model")
    parser.add_argument("--config_path", type=str, default="configs/base.yaml",
                        help="Path to config file")
    parser.add_argument("--random_seed", type=int, default=0, 
                        help="Random seed")
    
    # 缓存加速相关参数
    parser.add_argument("--cache_mode", type=str, default=None, 
                        choices=[None, 'threshold', 'optimal'], 
                        help="Cache acceleration mode")
    parser.add_argument("--cache_threshold", type=int, default=5, 
                        help="Cache threshold for threshold mode")
    parser.add_argument("--optimal_steps_dir", type=str, default=None, 
                        help="Directory containing optimal steps")
    parser.add_argument("--num_caches", type=int, default=5, 
                        help="Number of caches for optimal mode")
    parser.add_argument("--cache_metric", type=str, default='cosine', 
                        help="Metric for optimal steps files")
    parser.add_argument("--num_bu_blocks", type=int, default=0, 
                        help="Number of blocks to apply BU algorithm (0 to disable)")
    
    # 输出和测试参数
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--skip_video", action="store_true",
                        help="Skip video rendering")
    parser.add_argument("--skip_env_test", action="store_true",
                        help="Skip environment testing, only run performance benchmark")
    parser.add_argument("--benchmark_only", action="store_true",
                        help="Only run performance benchmark")
    
    return parser.parse_args(args)

def set_global_seed(seed):
    """设置全局随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dummy_input_for_benchmark(device='cuda:0', batch_size=1):
    """为性能测试创建虚拟输入"""
    # 模拟观察历史窗口
    obs_window = deque(maxlen=2)
    for _ in range(2):
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        obs_window.append(dummy_img)
    
    # 本体感知数据 - ManiSkill需要8维（7个手臂关节 + 1个夹爪）
    proprio = torch.randn(batch_size, 8, device=device)
    
    # 处理图像序列
    image_arrs = []
    for window_img in obs_window:
        image_arrs.append(window_img)
        image_arrs.append(None)
        image_arrs.append(None)
    
    images = [Image.fromarray(arr) if arr is not None else None for arr in image_arrs]
    
    return proprio, images

def run_performance_benchmark(original_policy, fast_policy, text_embed, device='cuda:0', num_trials=10):
    """运行性能基准测试，比较原始策略和缓存策略"""
    logger.info(f"=== 性能基准测试 (试验次数: {num_trials}) ===")
    
    # 预热
    logger.info("预热中...")
    proprio, images = create_dummy_input_for_benchmark(device)
    with torch.no_grad():
        for _ in range(3):
            # 预热原始策略
            if hasattr(original_policy, 'reset_cache'):
                original_policy.reset_cache()
            original_policy.step(proprio, images, text_embed)
            
            # 预热缓存策略
            if hasattr(fast_policy, 'reset_cache'):
                fast_policy.reset_cache()
            fast_policy.step(proprio, images, text_embed)
    
    # 测试原始策略
    logger.info("\n----- 原始策略 -----")
    original_durations = []
    for i in range(num_trials):
        if hasattr(original_policy, 'reset_cache'):
            original_policy.reset_cache()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            actions = original_policy.step(proprio, images, text_embed)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        duration = time.time() - start_time
        original_durations.append(duration)
        logger.info(f"运行 {i+1}/{num_trials}: {duration:.4f} 秒")
    
    # 计算原始策略统计数据
    avg_original = np.mean(original_durations)
    std_original = np.std(original_durations)
    frequency_original = 1.0 / avg_original
    logger.info(f"平均时间: {avg_original:.4f} ± {std_original:.4f} 秒")
    logger.info(f"推理频率: {frequency_original:.2f} FPS")
    
    # 测试缓存策略
    logger.info("\n----- 缓存策略 -----")
    fast_durations = []
    for i in range(num_trials):
        if hasattr(fast_policy, 'reset_cache'):
            fast_policy.reset_cache()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            actions = fast_policy.step(proprio, images, text_embed)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        duration = time.time() - start_time
        fast_durations.append(duration)
        logger.info(f"运行 {i+1}/{num_trials}: {duration:.4f} 秒")
    
    # 计算缓存策略统计数据
    avg_fast = np.mean(fast_durations)
    std_fast = np.std(fast_durations)
    frequency_fast = 1.0 / avg_fast
    speedup_time = avg_original / avg_fast
    
    logger.info(f"平均时间: {avg_fast:.4f} ± {std_fast:.4f} 秒")
    logger.info(f"推理频率: {frequency_fast:.2f} FPS")
    logger.info(f"加速比: {speedup_time:.2f}x")
    
    return {
        'original': {
            'avg_time': avg_original,
            'std_time': std_original,
            'frequency': frequency_original,
            'durations': original_durations
        },
        'fast': {
            'avg_time': avg_fast,
            'std_time': std_fast,
            'frequency': frequency_fast,
            'durations': fast_durations
        },
        'speedup': speedup_time
    }

def run_environment_evaluation(args, policy, text_embed):
    """运行环境评估"""
    logger.info(f"开始环境评估: {args.env_id}")
    
    # 创建环境
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        reward_mode=args.reward_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend
    )
    
    MAX_EPISODE_STEPS = 400
    total_episodes = args.num_traj
    success_count = 0
    
    base_seed = 20241201

    for episode in tqdm.trange(total_episodes):
        obs_window = deque(maxlen=2)
        obs, _ = env.reset(seed = episode + base_seed)
        policy.reset()

        img = env.render().squeeze(0).detach().cpu().numpy()
        obs_window.append(None)
        obs_window.append(np.array(img))
        proprio = obs['agent']['qpos'][:, :-1]

        global_steps = 0
        video_frames = []

        success_time = 0
        done = False

        while global_steps < MAX_EPISODE_STEPS and not done:
            image_arrs = []
            for window_img in obs_window:
                image_arrs.append(window_img)
                image_arrs.append(None)
                image_arrs.append(None)
            images = [Image.fromarray(arr) if arr is not None else None
                        for arr in image_arrs]
            actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()
            # Take 8 steps since RDT is trained to predict interpolated 64 steps(actual 14 steps)
            actions = actions[::4, :]
            for idx in range(actions.shape[0]):
                action = actions[idx]
                obs, reward, terminated, truncated, info = env.step(action)
                img = env.render().squeeze(0).detach().cpu().numpy()
                obs_window.append(img)
                proprio = obs['agent']['qpos'][:, :-1]
                video_frames.append(img)
                global_steps += 1
                if terminated or truncated:
                    assert "success" in info, sorted(info.keys())
                    if info['success']:
                        success_count += 1
                        done = True
                        break 
        
        # 记录每个episode的结果
        print(f"Trial {episode+1} finished, success: {info['success']}, steps: {global_steps}")

    success_rate = success_count / total_episodes * 100
    logger.info(f"最终成功率: {success_rate:.2f}% ({success_count}/{total_episodes})")
    
    env.close()
    
    return {
        'success_rate': success_rate,
        'success_count': success_count,
        'total_episodes': total_episodes
    }

def main():
    args = parse_args()
    
    # 设置随机种子
    set_global_seed(args.random_seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=== RDT缓存加速评估 ===")
    logger.info(f"环境: {args.env_id}")
    logger.info(f"缓存模式: {args.cache_mode}")
    logger.info(f"输出目录: {output_dir}")
    
    # 加载配置
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    
    # 创建原始模型
    logger.info("加载模型中...")
    original_policy = create_model(
        args=config,
        dtype=torch.bfloat16,
        pretrained=args.pretrained_path,
        pretrained_text_encoder_name_or_path="google/t5-v1_1-xxl",
        pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384"
    )
    
    # 创建缓存策略副本
    logger.info("创建策略副本用于缓存加速...")
    fast_policy = deepcopy(original_policy)
    
    # 应用缓存加速到副本
    if args.cache_mode is not None:
        logger.info(f"应用缓存加速: {args.cache_mode}")
        
        RDTCacheAccelerator.apply_cache(
            fast_policy.policy,
            cache_threshold=args.cache_threshold,
            optimal_steps_dir=args.optimal_steps_dir,
            num_caches=args.num_caches,
            metric=args.cache_metric,
            cache_mode=args.cache_mode,
            num_bu_blocks=args.num_bu_blocks
        )
        logger.info(f"缓存加速已应用")
    else:
        logger.info("未应用缓存加速，使用原始推理模式")
    
    # 准备语言指令
    task2lang = {
        "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
        "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
        "StackCube-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
        "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
        "PushCube-v1": "Push and move a cube to a goal region in front of it."
    }
    
    text_embed_cache_path = output_dir / f'text_embed_{args.env_id}.pt'
    if text_embed_cache_path.exists():
        text_embed = torch.load(text_embed_cache_path)
        logger.info("已加载缓存的文本嵌入")
    else:
        instruction = task2lang[args.env_id]
        text_embed = original_policy.encode_instruction(instruction)
        torch.save(text_embed, text_embed_cache_path)
        logger.info("已编码并缓存文本嵌入")
    
    # 运行性能基准测试（比较原始策略和缓存策略）
    benchmark_results = run_performance_benchmark(original_policy, fast_policy, text_embed)
    
    # 运行环境评估（如果需要）
    env_results = {}
    if not args.benchmark_only and not args.skip_env_test:
        # 使用缓存策略进行环境评估
        logger.info("\n使用缓存策略运行环境评估...")
        env_results = run_environment_evaluation(args, fast_policy, text_embed)
    
    # 保存结果
    results = {
        'config': {
            'env_id': args.env_id,
            'cache_mode': args.cache_mode,
            'cache_threshold': args.cache_threshold,
            'optimal_steps_dir': args.optimal_steps_dir,
            'num_caches': args.num_caches,
            'cache_metric': args.cache_metric,
            'pretrained_path': args.pretrained_path,
            'random_seed': args.random_seed,
            'num_bu_blocks': args.num_bu_blocks
        },
        'benchmark': benchmark_results,
        'environment': env_results
    }
    
    # 保存详细结果
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        # 处理numpy类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        import json
        json.dump(results, f, indent=2, default=convert_numpy)
    
    logger.info(f"评估结果已保存至: {results_path}")
    
    # 打印摘要
    logger.info("\n=== 评估摘要 ===")
    logger.info(f"平均推理时间: {benchmark_results['original']['avg_time']:.4f} 秒 → {benchmark_results['fast']['avg_time']:.4f} 秒")
    logger.info(f"推理频率: {benchmark_results['original']['frequency']:.2f} FPS → {benchmark_results['fast']['frequency']:.2f} FPS")
    logger.info(f"加速比: {benchmark_results['speedup']:.2f}x")
    
    if env_results:
        logger.info(f"成功率: {env_results['success_rate']:.2f}%")
        logger.info(f"成功次数: {env_results['success_count']}/{env_results['total_episodes']}")
    
    logger.info("评估完成!")

if __name__ == "__main__":
    main() 