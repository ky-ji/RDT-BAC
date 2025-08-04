#!/usr/bin/env python3

"""
RDT模型激活收集工具 - 简化版本

只收集原始模型的激活值，用于后续分析。

Usage:
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
    
python -m models.activation_utils.collect_activations_rdt -c /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt -o assets -t PickCube-v1

python -m models.activation_utils.collect_activations_rdt -c /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt -o assets -t PegInsertionSide-v1

python -m models.activation_utils.collect_activations_rdt -c /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt -o assets -t StackCube-v1

python -m models.activation_utils.collect_activations_rdt -c /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt -o assets -t PlugCharger-v1
python -m models.activation_utils.collect_activations_rdt -c /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt -o assets -t PushCube-v1
"""

import sys
import os
import torch
import logging
import pickle
from pathlib import Path
import numpy as np
import random
import torch.backends.cudnn as cudnn
import click
import types
import torch.nn as nn

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 设置环境变量以启用确定性算法
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# 全局默认随机种子
DEFAULT_RANDOM_SEED = 11

def set_global_seed(seed=DEFAULT_RANDOM_SEED):
    """设置所有随机数生成器的种子"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        cudnn.deterministic = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActivationCollector:
    """
    激活收集器，用于收集模型在推理过程中的激活值
    """
    
    def __init__(self):
        self.activations = {}
        self.hooks = []
        self.current_timestep = -1
        self.tracked_modules = set()
        self.modules_seen_this_step = set()
        self.last_activations = {}
        
    def _hook_fn(self, name):
        def hook(module, input, output):
            # 记录此模块在当前时间步被调用
            self.modules_seen_this_step.add(name)
            
            if isinstance(output, torch.Tensor):
                if name not in self.activations:
                    self.activations[name] = []
                # 转换为float32并转为numpy数组以避免BFloat16问题
                activation = output.detach().cpu().float().numpy()
                self.activations[name].append(activation)
                # 存储最新的激活值
                self.last_activations[name] = activation
            # 处理多头注意力输出
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                if name not in self.activations:
                    self.activations[name] = []
                # 转换为float32并转为numpy数组以避免BFloat16问题
                activation = output[0].detach().cpu().float().numpy()
                self.activations[name].append(activation)
                # 存储最新的激活值
                self.last_activations[name] = activation
        return hook
        
    def register_hooks(self, model):
        """注册钩子到关键的RDT组件"""
        logger.info("开始扫描RDT模型结构...")
        
        hook_count = 0
        component_count = 0
        
        # 定义关键的RDT组件名称模式
        key_component_patterns = [
            '.attn',         # self-attention组件
            '.cross_attn',   # cross-attention组件  
            '.ffn'           # FFN组件
        ]
        
        # 导入RDT相关的类来检查模块类型
        from timm.models.vision_transformer import Attention, Mlp
        from models.rdt.blocks import CrossAttention
        
        for name, module in model.named_modules():
            # 检查是否是我们关注的关键组件
            should_register = False
            for pattern in key_component_patterns:
                if name.endswith(pattern):
                    # 进一步检查模块类型以确保正确性
                    if (pattern == '.attn' and isinstance(module, Attention)) or \
                        (pattern == '.cross_attn' and isinstance(module, CrossAttention)) or \
                        (pattern == '.ffn' and isinstance(module, Mlp)):
                            should_register = True
                            component_count += 1
                            logger.info(f"找到关键RDT组件: {name} (类型: {type(module).__name__})")
                            break
            
            if should_register:
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
                hook_count += 1
                self.tracked_modules.add(name)
        
        logger.info(f"注册了 {hook_count} 个关键RDT组件钩子 (attn: {component_count//3}, cross_attn: {component_count//3}, ffn: {component_count//3})")
    
    def set_timestep(self, timestep):
        if self.current_timestep != -1 and timestep != self.current_timestep:
            self.handle_step_completion()
        
        self.current_timestep = timestep
        self.modules_seen_this_step = set()
    
    def handle_step_completion(self):
        """检查时间步结束时的缺失激活并用之前的值填充"""
        missing_modules = self.tracked_modules - self.modules_seen_this_step
        
        if missing_modules:
            logger.debug(f"时间步 {self.current_timestep}: {len(missing_modules)} 个模块缺失激活")
            
            for module_name in missing_modules:
                # 只有当我们有之前的值时才填充缺失的激活
                if module_name in self.last_activations:
                    if module_name not in self.activations:
                        self.activations[module_name] = []
                    
                    # 添加此模块的最新激活值（已经是numpy数组）
                    self.activations[module_name].append(self.last_activations[module_name])
                    logger.debug(f"  - 复用了 {module_name} 的之前激活")
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def get_activations(self):
        """返回收集的激活"""
        return self.activations
    
    def save_activations(self, save_path):
        """保存激活到文件"""
        with open(save_path, 'wb') as f:
            pickle.dump(self.activations, f)
        logger.info(f"保存了 {len(self.activations)} 个模块的激活到 {save_path}")

def collect_rdt_activations(
    checkpoint,
    output_base_dir,
    task_name='PickCube-v1',
    device='cuda:0',
    demo_idx=0,
    force_recompute=False,
    random_seed=DEFAULT_RANDOM_SEED
):
    """
    收集RDT模型原始激活值
    
    Args:
        checkpoint: RDT模型检查点路径
        output_base_dir: 输出基础目录
        task_name: 任务名称
        device: 运行设备
        demo_idx: 演示索引（用于文件命名）
        force_recompute: 是否强制重新计算
        random_seed: 随机种子
    """
    # 设置随机种子
    set_global_seed(random_seed)
    
    # 构建输出路径
    output_dir = Path(output_base_dir) / task_name / 'original'
    output_dir.mkdir(parents=True, exist_ok=True)
    activations_path = output_dir / 'activations.pkl'
    
    # 检查是否已存在激活文件
    if activations_path.exists() and not force_recompute:
        logger.info(f"发现现有激活文件: {activations_path}")
        try:
            with open(activations_path, 'rb') as f:
                activations = pickle.load(f)
            logger.info("成功加载现有激活")
            return activations
        except Exception as e:
            logger.warning(f"加载现有文件错误: {e}")
            logger.info("将重新计算激活")
    
    logger.info("开始计算RDT原始激活...")
    
    # 使用统一的策略运行接口
    from models.activation_utils.run_policy import run_policy
    
    # 加载模型和准备输入数据
    policy, obs_dict = run_policy(
        checkpoint=checkpoint,
        output_dir=str(output_dir),
        device=device,
        demo_idx=demo_idx,
        cache_mode='original',
        random_seed=random_seed
    )
    
    # 从obs_dict中提取数据
    proprio = obs_dict['proprio']
    images = obs_dict['images']
    text_embed = obs_dict['text_embed']
    
    # 找到实际的RDT模型
    if hasattr(policy, 'policy'):
        # 如果是包装后的policy
        actual_policy = policy.policy
        model = actual_policy.model
    elif hasattr(policy, 'model'):
        # 如果是直接的policy
        actual_policy = policy
        model = policy.model
    else:
        logger.error("无法找到模型，策略结构不正确")
        raise RuntimeError("Cannot find model in policy")
    
    # 创建激活收集器
    collector = ActivationCollector()
    collector.register_hooks(model)
    
    # 修补模型的forward方法以跟踪时间步
    original_model_forward = model.forward
    
    def patched_forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None):
        # 在收集器中跟踪当前时间步
        timestep_value = t.item() if hasattr(t, 'item') else t
        if isinstance(timestep_value, torch.Tensor):
            timestep_value = timestep_value.item()
        collector.set_timestep(timestep_value)
        return self._original_forward(x, freq, t, lang_c, img_c, lang_mask, img_mask)
    
    # 应用补丁
    model._original_forward = original_model_forward
    model.forward = types.MethodType(patched_forward, model)
    
    # 开始收集激活
    logger.info("开始收集激活...")
    with torch.no_grad():
        # 执行策略预测以触发完整的扩散过程
        if hasattr(policy, 'step'):
            action_dict = policy.step(proprio, images, text_embed)
        else:
            logger.warning("策略没有step方法，尝试使用其他预测方法")
            action_dict = None
    
    # 确保处理最后一个时间步
    collector.handle_step_completion()
    
    # 保存激活
    collector.save_activations(str(activations_path))
    
    # 移除钩子并恢复原始前向方法
    collector.remove_hooks()
    model.forward = original_model_forward
    
    logger.info(f"激活收集完成，保存至: {activations_path}")
    return collector.get_activations()

@click.command()
@click.option('-c', '--checkpoint', required=True, help='RDT模型检查点路径')
@click.option('-o', '--output_base_dir', required=True, help='输出基础目录')
@click.option('-t', '--task_name', default='PickCube-v1', help='任务名称')
@click.option('-d', '--device', default='cuda:0', help='运行设备')
@click.option('--demo_idx', default=0, type=int, help='演示索引')
@click.option('--force', is_flag=True, help='强制重新计算')
@click.option('--random_seed', default=DEFAULT_RANDOM_SEED, type=int, help='随机种子')
def main(checkpoint, output_base_dir, task_name, device, demo_idx, force, random_seed):
    """RDT激活收集命令行工具 - 只收集原始模式激活"""
    
    # 设置全局随机种子
    set_global_seed(random_seed)
    
    collect_rdt_activations(
        checkpoint=checkpoint,
        output_base_dir=output_base_dir,
        task_name=task_name,
        device=device,
        demo_idx=demo_idx,
        force_recompute=force,
        random_seed=random_seed
    )

if __name__ == '__main__':
    main() 