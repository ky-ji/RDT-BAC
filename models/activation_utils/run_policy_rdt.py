#!/usr/bin/env python3

"""
为RDT模型适配的策略运行器
参考Fast_diffusion_policy的run_policy.py实现

使用示例:
# 运行原始RDT模型
python -m models.activation_utils.run_policy_rdt -c /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt -o output -d cuda:0 --task_name PickCube-v1

# 运行带缓存的RDT模型
python -m models.activation_utils.run_policy_rdt -c /home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt -o output -d cuda:0 --task_name PickCube-v1 --cache_mode threshold --cache_threshold 10
"""

import sys
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from collections import deque
from PIL import Image
import click
import logging
import yaml
import gymnasium as gym

# 导入ManiSkill以注册环境
try:
    import mani_skill.envs  # 这会注册所有ManiSkill环境
except ImportError:
    print("Warning: mani_skill not found. ManiSkill环境将不可用。")
    mani_skill = None

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from models.rdt_runner import RDTRunner
from scripts.maniskill_model import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置环境变量以启用确定性算法
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# 全局默认随机种子
DEFAULT_RANDOM_SEED = 11

def set_seed_for_policy(seed=DEFAULT_RANDOM_SEED):
    """设置随机种子"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        cudnn.deterministic = True

def create_maniskill_input_data(task_name='PickCube-v1', device='cuda:0', random_seed=DEFAULT_RANDOM_SEED):
    """
    使用ManiSkill环境获取真实的输入数据
    完全参考eval_rdt_maniskill.py的实现
    """
    # 设置随机种子，确保环境初始化的一致性
    set_seed_for_policy(random_seed)
    
    # 创建ManiSkill环境，完全参考eval_rdt_maniskill.py
    env = gym.make(
        task_name,
        obs_mode="rgb",  # 使用RGB观察模式
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        sim_backend="auto"
    )
    
    # 创建观察窗口，参考eval_rdt_maniskill.py
    obs_window = deque(maxlen=2)
    
    # 重置环境并获取初始观察
    base_seed = 20241201  # 参考eval_rdt_maniskill.py中的base_seed
    obs, _ = env.reset(seed=base_seed)
    
    # 获取初始图像，参考eval_rdt_maniskill.py的实现
    img = env.render().squeeze(0).detach().cpu().numpy()
    
    # 初始化观察窗口，参考eval_rdt_maniskill.py
    obs_window.append(None)  # 第一帧为None
    obs_window.append(np.array(img))  # 第二帧为当前图像
    
    # 获取本体感知数据，参考eval_rdt_maniskill.py中的qpos处理
    # obs['agent']['qpos'][:, :-1] 表示取除了最后一维的所有关节位置
    proprio = obs['agent']['qpos'][:, :-1]  # 形状应该是(1, 7)
    
    # 转换为所需的设备和数据类型
    proprio = torch.tensor(proprio, device=device, dtype=torch.float32)
    
    # 处理图像数据，按RDT期望的格式组织
    # RDT期望：[ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1}, ext_{t}, right_wrist_{t}, left_wrist_{t}]
    # 但由于ManiSkill环境只有外部相机，我们需要创建适当的图像数组
    
    # 获取有效图像
    valid_images = [img for img in obs_window if img is not None]
    
    if len(valid_images) == 0:
        raise RuntimeError("No valid images found in observation window")
    
    # 创建图像数组，使用有效图像填充所有位置
    # 如果只有一个有效图像，就复制它；如果有两个，就重复使用它们
    image_arrs = []
    if len(valid_images) == 1:
        # 只有一个图像，复制6次
        img = valid_images[0]
        image_arrs = [img] * 6
    else:
        # 有两个图像，按时间顺序排列
        prev_img = valid_images[0] if obs_window[0] is not None else valid_images[-1]
        curr_img = valid_images[-1]
        # [ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1}, ext_{t}, right_wrist_{t}, left_wrist_{t}]
        image_arrs = [prev_img, prev_img, prev_img, curr_img, curr_img, curr_img]
    
    # 将numpy数组转换为PIL图像
    # 注意：图像预处理（包括resize到384x384）会在model.step()中由image_processor自动处理
    images = [Image.fromarray(arr) for arr in image_arrs]
    
    # 关闭环境以释放资源
    env.close()
    
    logger.info(f"Successfully created ManiSkill input data for {task_name}")
    logger.info(f"Proprio shape: {proprio.shape}")
    logger.info(f"Number of images: {len([img for img in images if img is not None])}")
    
    return proprio, images

def run_rdt_policy(
    checkpoint,
    output_dir,
    device,
    demo_idx=0,
    task_name='PickCube-v1',
    cache_mode='original',
    cache_threshold=5,
    optimal_steps_dir=None,
    num_caches=30,
    metric='cosine',
    num_bu_blocks=3,
    edit_steps=None,
    interpolation_ratio=1.0,
    reference_activations_path=None,
    return_obs_action=False,
    random_seed=DEFAULT_RANDOM_SEED
):
    """
    运行RDT策略模型并返回模型、输入和可选的输出
    参考Fast_diffusion_policy的run_policy实现
    
    Args:
        checkpoint: RDT模型检查点路径
        output_dir: 输出目录
        device: 运行设备
        demo_idx: 演示索引
        task_name: 任务名称
        cache_mode: 缓存模式 ('original', 'threshold', 'optimal', 'fix', 'propagate', 'edit')
        cache_threshold: 缓存阈值，每隔多少步更新一次缓存
        optimal_steps_dir: 最优步骤目录路径
        num_caches: 缓存更新次数
        metric: 相似度指标类型
        num_bu_blocks: 要应用BU算法的块数量，为0时禁用BU算法
        edit_steps: edit模式下使用的自定义steps
        interpolation_ratio: 插值比例
        reference_activations_path: 参考激活路径
        return_obs_action: 是否返回观察和动作
        random_seed: 随机种子，确保结果可重复
        
    Returns:
        policy: 策略模型
        obs_dict: 观察数据字典（包含proprio和images）
        action_dict: 如果return_obs_action为True则返回动作数据
    """
    # 设置随机种子
    set_seed_for_policy(random_seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 根据参考文件的实现加载RDT模型
    logger.info(f"Loading RDT model from: {checkpoint}")
    
    # 加载配置文件，参考eval_rdt_maniskill.py
    config_path = 'configs/base.yaml'
    # 动态获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    config_full_path = os.path.join(project_root, config_path)
    
    if os.path.exists(config_full_path):
        with open(config_full_path, "r") as fp:
            config = yaml.safe_load(fp)
    else:
        logger.warning(f"Config file not found at {config_full_path}, using default config")
        # 使用默认配置，添加dataset部分
        config = {
            'action_dim': 7,
            'pred_horizon': 64,
            'lang_token_dim': 4096,
            'img_token_dim': 1152,
            'state_token_dim': 7,
            'max_lang_cond_len': 77,
            'img_cond_len': 6,
            'dataset': {
                'tokenizer_max_length': 1024
            }
        }
    
    # 设置预训练编码器路径，参考eval_rdt_maniskill.py
    pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    
    # 使用create_model函数创建模型，参考eval_rdt_maniskill.py
    # 添加image_size参数以确保图像预处理正确
    policy = create_model(
        args=config, 
        device=device,
        dtype=torch.bfloat16,
        pretrained=checkpoint,
        pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        image_size=(384, 384)  # SigLIP期望的图像尺寸
    )
    
    device = torch.device(device)
    # RoboticDiffusionTransformerModel已经在初始化时设置了设备
    # policy = policy.to(device)  # 这行会导致错误，因为RoboticDiffusionTransformerModel没有to方法
    # policy.eval()  # reset()方法已经设置为eval模式
    
    # 应用缓存加速（如果RDT支持的话）
    if cache_mode != 'original':
        logger.info(f"应用{cache_mode}模式缓存加速")
        
        if cache_mode in ['optimal', 'fix']:
            logger.info(f"最优步骤目录: {optimal_steps_dir}, 指标: {metric}, 缓存数: {num_caches}")
        elif cache_mode == 'threshold':
            logger.info(f"缓存阈值: {cache_threshold}")
        elif cache_mode == 'edit':
            logger.info(f"Edit模式自定义steps: {edit_steps}")
        
        if num_bu_blocks > 0:
            logger.info(f"使用BU算法优化{num_bu_blocks}个块")
        
        if interpolation_ratio < 1.0:
            logger.info(f"使用插值比例: {interpolation_ratio}")
            if reference_activations_path:
                logger.info(f"参考激活路径: {reference_activations_path}")
        
        # 注意：RDT的缓存机制可能与FastDiffusionPolicy不同
        # 这里需要根据RDT的实际实现来调整
        logger.warning("RDT缓存模式尚未完全实现，使用原始模式")
    else:
        logger.info("使用原始模式，不应用缓存加速")
    
    # 重置策略
    if hasattr(policy, 'reset'):
        policy.reset()
    
    # 如果使用缓存，确保缓存策略被重置
    if cache_mode != 'original' and hasattr(policy, 'reset_cache'):
        policy.reset_cache()
    
    # 创建真实输入数据，使用ManiSkill环境
    proprio, images = create_maniskill_input_data(task_name, device, random_seed)
    
    # 创建语言指令嵌入，参考eval_rdt_maniskill.py中的task2lang
    task2lang = {
        "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
        "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
        "StackCube-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
        "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
        "PushCube-v1": "Push and move a cube to a goal region in front of it."
    }
    
    instruction = task2lang.get(task_name, 'Complete the task.')
    logger.info(f"Using instruction for {task_name}: {instruction}")
    
    # 检查是否存在预计算的text embedding
    text_embed_path = f'text_embed_{task_name}.pt'
    text_embed_full_path = os.path.join(project_root, text_embed_path)
    
    if os.path.exists(text_embed_full_path):
        logger.info(f"Loading pre-computed text embedding from {text_embed_full_path}")
        text_embed = torch.load(text_embed_full_path, map_location=device)
    else:
        # 使用策略的语言编码器获取嵌入
        logger.info("Computing text embedding using policy encoder")
        if hasattr(policy, 'encode_instruction'):
            text_embed = policy.encode_instruction(instruction)
            # 保存计算的嵌入以便后续使用
            torch.save(text_embed, text_embed_full_path)
            logger.info(f"Saved text embedding to {text_embed_full_path}")
        else:
            # 如果没有语言编码器，创建虚拟嵌入
            logger.error("策略没有语言编码器，无法生成text embedding")
            raise RuntimeError("Policy does not have encode_instruction method")
    
    # 构造观察数据字典（为了与Fast_diffusion_policy的接口保持一致）
    obs_dict = {
        'proprio': proprio,
        'images': images,
        'text_embed': text_embed
    }
    
    # 如果需要返回动作，执行预测
    if return_obs_action:
        with torch.no_grad():
            # 调用策略的step方法，参考eval_rdt_maniskill.py中的实现
            action_dict = policy.step(proprio, images, text_embed)
            
            # 处理动作输出，确保格式正确
            if isinstance(action_dict, torch.Tensor):
                # 如果直接返回tensor，保持原样
                pass
            elif hasattr(action_dict, 'squeeze'):
                # 如果有squeeze方法，去除多余维度
                action_dict = action_dict.squeeze(0)
            else:
                logger.warning(f"Unexpected action_dict type: {type(action_dict)}")
            
        return policy, obs_dict, action_dict
    
    return policy, obs_dict

@click.command()
@click.option('-c', '--checkpoint', required=True, help='RDT模型检查点路径')
@click.option('-o', '--output_dir', required=True, help='输出目录')
@click.option('-d', '--device', default='cuda:0', help='运行设备')
@click.option('--demo_idx', default=0, type=int, help='演示索引')
@click.option('--task_name', default='PickCube-v1', help='任务名称')
@click.option('--cache_mode', default='original', 
              type=click.Choice(['original', 'threshold', 'optimal', 'fix', 'propagate', 'edit']), 
              help='缓存模式')
@click.option('--cache_threshold', default=5, type=int, help='缓存阈值，每隔多少步更新一次缓存')
@click.option('--optimal_steps_dir', default=None, help='最优步骤目录路径')
@click.option('--num_caches', default=30, type=int, help='缓存更新次数')
@click.option('--metric', default='cosine', help='相似度指标类型，用于加载最优步骤文件')
@click.option('--num_bu_blocks', default=3, type=int, help='要应用BU算法的块数量，为0时禁用BU算法')
@click.option('--edit_steps', default=None, help='Edit模式自定义steps (逗号分隔的整数，例如 "0,20,40,60,80")')
@click.option('--interpolation_ratio', default=1.0, type=float, help='插值比例 (default=1.0)')
@click.option('--reference_activations_path', default=None, help='参考激活路径')
@click.option('--random_seed', default=DEFAULT_RANDOM_SEED, type=int, help='随机种子')
def main(checkpoint, output_dir, device, demo_idx, task_name,
         cache_mode, cache_threshold, optimal_steps_dir, 
         num_caches, metric, num_bu_blocks, edit_steps,
         interpolation_ratio, reference_activations_path, random_seed):
    """RDT策略运行器命令行工具"""
    
    # 处理edit_steps参数
    processed_edit_steps = None
    if edit_steps is not None:
        try:
            processed_edit_steps = [int(x.strip()) for x in edit_steps.split(',')]
            logger.info(f"解析edit steps: {processed_edit_steps}")
        except Exception as e:
            logger.error(f"解析edit_steps '{edit_steps}'出错: {e}")
            logger.error("格式应为逗号分隔的整数，例如 '0,20,40,60,80'")
            return
    
    policy, obs_dict, action_dict = run_rdt_policy(
        checkpoint, output_dir, device, demo_idx, task_name,
        cache_mode, cache_threshold, optimal_steps_dir,
        num_caches, metric, num_bu_blocks,
        processed_edit_steps, interpolation_ratio, reference_activations_path,
        return_obs_action=True,
        random_seed=random_seed
    )
    
    # 转回CPU并转换为numpy
    if isinstance(action_dict, torch.Tensor):
        action = action_dict.detach().to('cpu').numpy()
    elif isinstance(action_dict, dict):
        action = action_dict.get('action', action_dict.get('actions'))
        if isinstance(action, torch.Tensor):
            action = action.detach().to('cpu').numpy()
    else:
        action = action_dict
    
    if not np.all(np.isfinite(action)):
        raise RuntimeError("Nan or Inf action")
    
    logger.info(f"Action shape: {action.shape}")
    logger.info(f"Action: {action}")
    
    return action

if __name__ == '__main__':
    main() 