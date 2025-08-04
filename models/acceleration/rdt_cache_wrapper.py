import torch
import types
import logging
import os
import pickle
import numpy as np
import random
import re
from typing import List, Dict, Optional, Any, Union, Tuple
from pathlib import Path
import torch.nn as nn

logger = logging.getLogger(__name__)

class RDTCacheAccelerator:
    """
    RDT模型缓存加速器
    参考Fast_diffusion_policy实现，支持多种缓存模式和BU算法
    
    支持的缓存模式：
    0. original: 不使用缓存，直接返回原始策略
    1. threshold: 基于固定阈值的缓存
    2. optimal: 基于最优缓存步骤的缓存
    3. fix: 所有Block共享最后一组Block的optimal steps
    4. propagate: 第一个Block只在step 0计算，其他步骤使用缓存
    5. edit: 使用自定义steps
    """

    @staticmethod
    def apply_cache(
        policy,
        cache_threshold: int = 5,
        optimal_steps_dir: str = None,
        num_caches: int = 30,
        metric: str = 'cosine',
        cache_mode: str = None,
        interpolation_ratio: float = 1.0,
        reference_activations_path: str = None,
        edit_steps: List[int] = None,
        num_bu_blocks: int = 3
    ):
        """
        为RDT模型应用缓存加速，参考Fast_diffusion_policy实现
        
        Args:
            policy: RDT策略实例
            cache_threshold: 缓存阈值，每隔多少步更新一次缓存（threshold模式）
            optimal_steps_dir: 包含最优步骤的目录路径（optimal模式）
            num_caches: 缓存更新次数，用于加载对应的最优步骤文件
            metric: 指标类型，用于加载正确的最优步骤文件，默认'cosine'
            cache_mode: 指定缓存模式，可选 'original', 'threshold', 'optimal', 'fix', 'propagate', 'edit'
            interpolation_ratio: 插值比例，控制缓存激活与原始激活的混合比例
            reference_activations_path: 原始模式激活值路径，用于插值
            edit_steps: edit模式下使用的自定义steps列表
            num_bu_blocks: 要应用BU算法的块数量，为0时禁用BU算法
        
        Returns:
            更新后的策略实例（原地修改）或原始策略（original模式）
        """
        # 确定缓存模式
        if cache_mode is None:
            # 自动推断模式
            if optimal_steps_dir is not None:
                cache_mode = 'optimal'
            else:
                cache_mode = 'threshold'
        
        # 如果是原始模式，直接返回原策略
        if cache_mode == 'original':
            logger.info("使用原始模式，不应用缓存加速")
            return policy

        # 检查policy是否有model属性
        model = getattr(policy, 'model', None)
        if model is None:
            logger.error("策略必须有model属性")
            return policy
        
        # 检查model是否有blocks属性（RDT特有）
        if not hasattr(model, 'blocks'):
            logger.error("RDT模型必须有blocks属性")
            return policy
            
        num_inference_steps = getattr(policy, 'num_inference_timesteps', 100)

        # 检查插值比例参数
        if interpolation_ratio < 0.0 or interpolation_ratio > 1.0:
            logger.warning(f"插值比例必须在[0,1]范围内，收到的值为{interpolation_ratio}，将使用默认值1.0")
            interpolation_ratio = 1.0
        
        # 如果启用插值但没有提供参考激活路径，输出警告
        reference_activations = None
        if interpolation_ratio < 1.0 and reference_activations_path:
            logger.info(f"尝试加载参考激活值: {reference_activations_path}")
            try:
                with open(reference_activations_path, 'rb') as f:
                    reference_activations = pickle.load(f)
                
                # 打印参考激活的信息
                activation_keys = list(reference_activations.keys())
                logger.info(f"成功加载参考激活值，包含 {len(activation_keys)} 个模块")
                logger.info(f"前5个模块: {activation_keys[:5]}")
                
                # 随机选择一个模块查看内容
                if activation_keys:
                    sample_key = activation_keys[0]
                    sample_data = reference_activations[sample_key]
                    if isinstance(sample_data, list):
                        logger.info(f"模块 {sample_key} 包含 {len(sample_data)} 个时间步的激活值")
                        if len(sample_data) > 0:
                            logger.info(f"第一个时间步的形状: {sample_data[0].shape}")
                    else:
                        logger.info(f"模块 {sample_key} 的数据类型: {type(sample_data)}")
                
                logger.info(f"将使用插值比例: {interpolation_ratio}")
            except Exception as e:
                logger.error(f"加载参考激活值失败: {e}")
                logger.warning(f"将不使用插值，使用完整缓存")
                interpolation_ratio = 1.0
        elif interpolation_ratio < 1.0:
            logger.warning(f"启用了插值(ratio={interpolation_ratio})但未提供参考激活路径，将使用完整缓存")
            interpolation_ratio = 1.0

        # 创建缓存结构
        cache = {
            'threshold': cache_threshold,      # 缓存阈值
            'mode': cache_mode,               # 缓存模式
            'metric': metric,                 # 相似度指标
            'optimal_steps_dir': optimal_steps_dir, # 最优步骤目录
            'num_caches': num_caches,         # 缓存更新次数
            'num_steps': num_inference_steps, # 总步数
            'current_step': -1,               # 当前步数（从0开始）
            'block_cache': {},                # 块缓存，格式：{block_key: output}
            'block_steps': {},                # 每个块的步骤，格式：{block_key: [steps]}
            'num_bu_blocks': num_bu_blocks,   # 要应用BU算法的块数量
            'interpolation_ratio': interpolation_ratio,  # 插值比例
            'reference_activations': reference_activations,  # 参考激活值
            'edit_steps': edit_steps          # edit模式的自定义steps
        }
        policy._cache = cache

        # 找到所有需要缓存的RDT blocks
        cacheable_layers = RDTCacheAccelerator._find_cacheable_layers(model)
        policy._cacheable_layers = cacheable_layers
        logger.info(f"找到 {len(cacheable_layers)} 个需要缓存的RDT块")

        # 根据不同的缓存模式设置步骤
        if cache_mode == 'optimal' and optimal_steps_dir:
            # 基于最优步骤模式，加载每个块的最优步骤
            RDTCacheAccelerator._load_block_optimal_steps(cache, cacheable_layers, optimal_steps_dir, num_caches, metric)
            cache_threshold = None
        elif cache_mode == 'fix' and optimal_steps_dir:
            # Fix模式：所有Block共享最后一组Block的optimal steps
            RDTCacheAccelerator._setup_fix_mode(cache, cacheable_layers, optimal_steps_dir, num_caches, metric)
            cache_threshold = None
        elif cache_mode == 'propagate':
            # Propagate模式：第一个Block只计算step 0，后续Block全部计算
            RDTCacheAccelerator._setup_propagate_mode(cache, cacheable_layers, num_inference_steps)
            cache_threshold = None
        elif cache_mode == 'edit':
            # Edit模式：所有block共用一个通过参数指定的steps
            RDTCacheAccelerator._setup_edit_mode(cache, cacheable_layers)
            cache_threshold = None

        # 为每个层添加缓存功能
        for layer_name, layer in cacheable_layers:
            RDTCacheAccelerator._add_cache_to_block(layer, layer_name, cache)

        # 保存原始forward方法
        original_forward = model.forward

        # 创建带缓存功能的forward方法
        def forward_with_cache(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None):
            """使用缓存的forward方法"""
            cache = getattr(policy, '_cache', None)

            # 增加步数计数器
            if cache is not None:
                cache['current_step'] += 1
                current_step = cache['current_step']

                # 为所有块设置当前步骤和缓存标志
                RDTCacheAccelerator._update_cache_flags(cache, current_step)

            # 执行前向传播
            output = original_forward(x, freq, t, lang_c, img_c, lang_mask, img_mask)

            return output

        # 替换模型的forward方法
        model.forward = types.MethodType(forward_with_cache, model)

        # 添加重置缓存方法
        def reset_cache(self):
            """重置缓存"""
            if hasattr(self, '_cache'):
                self._cache['current_step'] = -1
                self._cache['block_cache'] = {}
                logger.debug("缓存已重置")
            return self

        # 添加方法到策略对象
        policy.reset_cache = types.MethodType(reset_cache, policy)

        # 保存原始的predict_action方法（如果存在）
        if hasattr(policy, 'predict_action'):
            original_predict_action = policy.predict_action

            # 创建带自动重置缓存的predict_action方法
            def predict_action_with_auto_reset(self, *args, **kwargs):
                """在每次predict_action调用前自动重置缓存"""
                # 调用重置缓存方法
                self.reset_cache()
                # 调用原始的predict_action方法
                return original_predict_action(*args, **kwargs)

            # 替换策略的predict_action方法
            policy.predict_action = types.MethodType(predict_action_with_auto_reset, policy)

        # 日志输出
        if cache_mode == 'threshold':
            logger.info(f"基于阈值的缓存加速已成功应用到策略，缓存阈值: {cache_threshold}")
        elif cache_mode == 'optimal':
            logger.info(f"基于最优步骤的缓存加速已成功应用到策略，步骤目录: {optimal_steps_dir}, 缓存次数: {num_caches}")
        elif cache_mode == 'fix':
            logger.info(f"Fix模式缓存加速已成功应用到策略，所有Block共享最后一层的optimal steps")
        elif cache_mode == 'propagate':
            logger.info(f"Propagate模式缓存加速已成功应用到策略，第一个Block只计算step 0")
        elif cache_mode == 'edit':
            logger.info(f"Edit模式缓存加速已成功应用到策略，所有block共用自定义steps: {edit_steps}")

        # 修改日志输出，现在由num_bu_blocks控制
        if num_bu_blocks > 0:
            logger.info(f"启用BU算法，将应用于误差最大的 {num_bu_blocks} 个块")
        else:
            logger.info(f"BU算法已禁用 (num_bu_blocks = {num_bu_blocks})")
        
        if interpolation_ratio < 1.0:
            logger.info(f"激活插值已启用: 使用{interpolation_ratio * 100:.1f}%的缓存激活 + {(1-interpolation_ratio) * 100:.1f}%的原始激活")

        return policy

    @staticmethod
    def _find_cacheable_layers(model) -> List[Tuple[str, Any]]:
        """
        找到模型中所有可缓存的RDT块
        
        Args:
            model: RDT模型
            
        Returns:
            可缓存层列表 [(name, layer), ...]
        """
        cacheable_layers = []
        
        # 对于RDT模型，缓存blocks中的每个块
        if hasattr(model, 'blocks'):
            for i, block in enumerate(model.blocks):
                name = f'blocks.{i}'
                cacheable_layers.append((name, block))
                logger.info(f"选择缓存RDT块: {name}")
        
        logger.info(f"共找到 {len(cacheable_layers)} 个可缓存的RDT块")
        return cacheable_layers

    @staticmethod
    def _load_block_optimal_steps(cache: Dict, layers: List[Tuple[str, Any]],
                                 optimal_steps_dir: str, num_caches: int, metric: str):
        """
        为每个RDT层的各个计算块加载最优缓存步骤
        
        针对每个RDT层, 为主要计算块分别加载步骤:
        - attn_block: 自注意力块
        - cross_attn_block: 交叉注意力块
        - ffn_block: 前馈网络块
        
        Args:
            cache: 缓存字典
            layers: 层列表 [(name, layer), ...]
            optimal_steps_dir: 最优步骤目录
            num_caches: 缓存更新次数
            metric: 相似度指标
        """
        steps_dir = Path(optimal_steps_dir)
        assert steps_dir.exists(), f"最优步骤目录 {optimal_steps_dir} 不存在"

        # 存储每个块的步骤
        block_steps = {}

        for layer_name, layer in layers:
            # 为层中的三个主要块查找对应的步骤文件
            attn_module_name = f"{layer_name}.attn"
            cross_attn_module_name = f"{layer_name}.cross_attn"
            ffn_module_name = f"{layer_name}.ffn"

            # attn_block使用attn模块的步骤
            attn_block_key = f"{layer_name}_attn_block"
            steps_file = steps_dir / metric / attn_module_name / f'optimal_steps_{attn_module_name}_{num_caches}_{metric}.pkl'
            if steps_file.exists():
                try:
                    with open(steps_file, 'rb') as f:
                        steps = pickle.load(f)
                    block_steps[attn_block_key] = steps
                    logger.info(f"为自注意力块 {attn_block_key} 加载步骤: {steps}")
                except Exception as e:
                    logger.warning(f"加载 {attn_block_key} 的步骤失败: {e}")

            # cross_attn_block使用cross_attn模块的步骤
            cross_attn_block_key = f"{layer_name}_cross_attn_block"
            steps_file = steps_dir / metric / cross_attn_module_name / f'optimal_steps_{cross_attn_module_name}_{num_caches}_{metric}.pkl'
            if steps_file.exists():
                try:
                    with open(steps_file, 'rb') as f:
                        steps = pickle.load(f)
                    block_steps[cross_attn_block_key] = steps
                    logger.info(f"为交叉注意力块 {cross_attn_block_key} 加载步骤: {steps}")
                except Exception as e:
                    logger.warning(f"加载 {cross_attn_block_key} 的步骤失败: {e}")

            # ffn_block使用ffn模块的步骤
            ffn_block_key = f"{layer_name}_ffn_block"
            steps_file = steps_dir / metric / ffn_module_name / f'optimal_steps_{ffn_module_name}_{num_caches}_{metric}.pkl'
            if steps_file.exists():
                try:
                    with open(steps_file, 'rb') as f:
                        steps = pickle.load(f)
                    block_steps[ffn_block_key] = steps
                    logger.info(f"为前馈网络块 {ffn_block_key} 加载步骤: {steps}")
                except Exception as e:
                    logger.warning(f"加载 {ffn_block_key} 的步骤失败: {e}")

        # 应用BU算法（如果启用）
        if cache.get('num_bu_blocks', 0) > 0:
            logger.info("Applying BU (Bottom-Up) steps propagation for RDT.")
            
            # 尝试从分析结果加载误差最大的块
            try:
                from models.activation_utils.bu_block_selection import (
                    analyze_block_errors, 
                    get_analysis_output_dir
                )
                
                # 使用optimal_steps_dir推导分析输出目录
                bu_blocks = []
                if optimal_steps_dir:
                    steps_path = Path(optimal_steps_dir)
                    # 例如：./assets/PickCube-v1/optimal_steps/cosine
                    # 我们需要得到：./assets/PickCube-v1/original/bu_block_selection
                    if "optimal_steps" in str(steps_path):
                        # 从 ./assets/PickCube-v1/optimal_steps/cosine 得到 ./assets/PickCube-v1
                        if steps_path.name == "cosine":
                            # 如果路径以cosine结尾，向上两级
                            task_base_path = steps_path.parent.parent
                        else:
                            # 否则向上一级
                            task_base_path = steps_path.parent
                        
                        # 构建bu_block_selection路径：./assets/PickCube-v1/original/bu_block_selection
                        bu_analysis_dir = task_base_path / "original" / "bu_block_selection"
                        logger.info(f"从optimal_steps_dir推导出BU分析目录: {bu_analysis_dir}")
                        
                        num_blocks = cache.get('num_bu_blocks', 3)
                        
                        # 检查BU分析目录是否存在
                        if bu_analysis_dir.exists():
                            logger.info(f"BU分析输出目录存在: {bu_analysis_dir}")
                            selected_path = bu_analysis_dir / f'top_{num_blocks}_error_blocks.pkl'
                            
                            if selected_path.exists():
                                # 从文件加载选定的块
                                with open(selected_path, 'rb') as f:
                                    selected_blocks_dict = pickle.load(f)
                                    bu_blocks = list(selected_blocks_dict.keys())
                                    logger.info(f"从文件加载了 {len(bu_blocks)} 个误差最大的块用于BU算法: {bu_blocks}")
                            else:
                                logger.warning(f"BU块选择文件不存在: {selected_path}")
                        else:
                            logger.warning(f"BU分析目录不存在: {bu_analysis_dir}")
                    else:
                        logger.warning(f"无法从optimal_steps_dir推导基础目录: {optimal_steps_dir}")
                else:
                    logger.warning(f"未提供optimal_steps_dir")
                
                # 应用BU算法传播
                if len(bu_blocks) > 1:
                    # 根据块名称排序，确保它们按照从浅到深的顺序
                    def get_layer_idx(block_key):
                        match = re.match(r'blocks\.(\d+)_([a-z_]+)', block_key)
                        if match:
                            return int(match.group(1))
                        return -1
                    
                    def get_block_type_priority(block_key):
                        match = re.match(r'blocks\.\d+_([a-z_]+)', block_key)
                        if match:
                            block_type = match.group(1)
                            priorities = {'attn_block': 0, 'cross_attn_block': 1, 'ffn_block': 2}
                            return priorities.get(block_type, 10)
                        return 10
                    
                    def get_block_type(block_key):
                        match = re.match(r'blocks\.\d+_([a-z_]+)', block_key)
                        if match:
                            return match.group(1)
                        return ""
                    
                    # 按层索引和块类型排序
                    sorted_bu_blocks = sorted(bu_blocks, key=lambda x: (get_layer_idx(x), get_block_type_priority(x)))
                    logger.info(f"BU: 排序后的块: {sorted_bu_blocks}")
                    
                    # 找出所有FFN块并按层索引排序
                    all_ffn_blocks = []
                    for block_key in block_steps.keys():
                        if get_block_type(block_key) == 'ffn_block':
                            all_ffn_blocks.append(block_key)
                    
                    sorted_all_ffn_blocks = sorted(all_ffn_blocks, key=lambda x: get_layer_idx(x))
                    logger.info(f"BU: 所有FFN块（按层索引排序）: {sorted_all_ffn_blocks}")
                    
                    # 为bu_blocks中的每个块从后层FFN Block获取steps
                    for block_key in sorted_bu_blocks:
                        block_layer_idx = get_layer_idx(block_key)
                        
                        # 收集所有更深层的FFN Block
                        deeper_ffn_blocks = []
                        for ffn_block in sorted_all_ffn_blocks:
                            ffn_layer_idx = get_layer_idx(ffn_block)
                            if ffn_layer_idx >= block_layer_idx:
                                deeper_ffn_blocks.append(ffn_block)
                        
                        if deeper_ffn_blocks:
                            logger.info(f"BU: 为块 {block_key} (层 {block_layer_idx}) 找到后层FFN块: {deeper_ffn_blocks}")
                            
                            # 合并所有后层FFN Block的steps
                            all_deeper_steps = set()
                            for deeper_ffn_block in deeper_ffn_blocks:
                                if deeper_ffn_block in block_steps:
                                    all_deeper_steps.update(block_steps[deeper_ffn_block])
                            
                            # 确保当前块具有所有后层FFN Block的steps
                            if block_key in block_steps and all_deeper_steps:
                                current_steps = set(block_steps[block_key])
                                missing_steps = all_deeper_steps - current_steps
                                
                                if missing_steps:
                                    updated_steps = sorted(list(current_steps.union(missing_steps)))
                                    block_steps[block_key] = updated_steps
                                    logger.info(f"BU: 为块 {block_key} 补充后层FFN Block的steps: {sorted(list(missing_steps))}。新steps: {updated_steps}")
                    
                    # 打印BU结果摘要
                    logger.info("\n===== BU算法应用结果摘要 =====")
                    logger.info(f"总共处理了 {len(bu_blocks)} 个BU块")
                    
                    # 统计每个块的steps数量
                    for block_key in sorted(bu_blocks, key=lambda x: (get_layer_idx(x), get_block_type_priority(x))):
                        if block_key in block_steps:
                            steps_count = len(block_steps[block_key])
                            logger.info(f"  块 {block_key}: {steps_count} 个steps")
                    
                    logger.info("BU算法：从所有后层FFN Block获取steps，确保bu_blocks中的块能获取所有必要的steps")
                    logger.info("===== BU算法结束 =====\n")
                    
                else:
                    logger.info("BU: 参与传播的块不足（至少需要2个）。")
                
                logger.info("BU算法应用完成")
                
            except Exception as e:
                logger.warning(f"应用BU算法时出错: {e}")
                import traceback
                logger.warning(f"详细错误信息: {traceback.format_exc()}")

        # 存储到缓存
        cache['block_steps'] = block_steps


        # 日志输出
        logger.info(f"共为 {len(block_steps)} 个计算块加载了最优步骤")

    @staticmethod
    def _setup_fix_mode(cache: Dict, layers: List[Tuple[str, Any]],
                        optimal_steps_dir: str, num_caches: int, metric: str):
        """
        Fix模式：所有Block共享最后一组Block的optimal steps
        
        Args:
            cache: 缓存字典
            layers: 层列表 [(name, layer), ...]
            optimal_steps_dir: 最优步骤目录
            num_caches: 缓存更新次数
            metric: 相似度指标
        """
        steps_dir = Path(optimal_steps_dir)
        assert steps_dir.exists(), f"最优步骤目录 {optimal_steps_dir} 不存在"

        # 存储每个块的步骤
        block_steps = {}
        
        # 提取所有层的索引和名称
        layer_indices = []
        for layer_name, _ in layers:
            # 从层名称中提取索引，例如从"blocks.2"中提取"2"
            parts = layer_name.split('.')
            if len(parts) >= 2 and parts[0] == 'blocks':
                try:
                    layer_idx = int(parts[1])
                    layer_indices.append((layer_idx, layer_name))
                except ValueError:
                    logger.warning(f"无法从层名称 {layer_name} 中提取索引")
        
        # 按索引排序层
        layer_indices.sort(key=lambda x: x[0])
        
        if not layer_indices:
            logger.warning("未找到有效的层索引，Fix模式无法应用")
            return
            
        # 获取最后一层的名称
        last_layer_idx, last_layer_name = layer_indices[-1]
        
        # 为最后一层的FFN Block加载最优步骤
        last_ffn_block_key = f"{last_layer_name}_ffn_block"
        last_ffn_module_name = f"{last_layer_name}.ffn"
        last_steps_file = steps_dir / metric / last_ffn_module_name / f'optimal_steps_{last_ffn_module_name}_{num_caches}_{metric}.pkl'
        
        # 确保最后一层的步骤文件存在
        if not last_steps_file.exists():
            logger.error(f"最后一层的步骤文件不存在: {last_steps_file}")
            return
            
        # 加载最后一层的最优步骤
        try:
            with open(last_steps_file, 'rb') as f:
                last_ffn_steps = pickle.load(f)
        except Exception as e:
            logger.error(f"加载最后一层FFN步骤失败: {e}")
            return
            
        logger.info(f"已加载最后一层 {last_layer_name} 的FFN最优步骤: {last_ffn_steps}")
        
        # 首先正常加载所有层的注意力块的步骤
        for layer_idx, layer_name in layer_indices:
            # 自注意力块
            attn_block_key = f"{layer_name}_attn_block"
            attn_module_name = f"{layer_name}.attn.dropout"
            steps_file = steps_dir / attn_module_name / f'optimal_steps_{attn_module_name}_{num_caches}_{metric}.pkl'
            if steps_file.exists():
                try:
                    with open(steps_file, 'rb') as f:
                        steps = pickle.load(f)
                    block_steps[attn_block_key] = steps
                    logger.info(f"为自注意力块 {attn_block_key} 加载步骤: {steps}")
                except Exception as e:
                    logger.warning(f"加载 {attn_block_key} 的步骤失败: {e}")
            
            # 交叉注意力块
            cross_attn_block_key = f"{layer_name}_cross_attn_block"
            cross_attn_module_name = f"{layer_name}.cross_attn.dropout"
            steps_file = steps_dir / cross_attn_module_name / f'optimal_steps_{cross_attn_module_name}_{num_caches}_{metric}.pkl'
            if steps_file.exists():
                try:
                    with open(steps_file, 'rb') as f:
                        steps = pickle.load(f)
                    block_steps[cross_attn_block_key] = steps
                    logger.info(f"为交叉注意力块 {cross_attn_block_key} 加载步骤: {steps}")
                except Exception as e:
                    logger.warning(f"加载 {cross_attn_block_key} 的步骤失败: {e}")
                    
            # 为所有层的FFN块使用最后一层的步骤
            ffn_block_key = f"{layer_name}_ffn_block"
            block_steps[ffn_block_key] = last_ffn_steps.copy()
            logger.info(f"为FFN块 {ffn_block_key} 分配最后一层的步骤: {last_ffn_steps}")
        
        # 确保至少找到了一些步骤
        assert len(block_steps) > 0, "未找到任何步骤，Fix模式无法应用"
        
        # 存储到缓存
        cache['block_steps'] = block_steps
        
        # 日志输出
        logger.info(f"Fix模式：已为所有FFN Block分配最后一层 {last_layer_name} 的最优步骤")

    @staticmethod
    def _setup_propagate_mode(cache: Dict, layers: List[Tuple[str, Any]], num_steps: int):
        """
        Propagate模式：第一个Block只计算step 0，后续Block全部计算
        
        Args:
            cache: 缓存字典
            layers: 层列表 [(name, layer), ...]
            num_steps: 总步骤数
        """
        # 存储每个块的步骤
        block_steps = {}
        
        # 提取所有层的索引和名称
        layer_indices = []
        for layer_name, _ in layers:
            # 从层名称中提取索引，例如从"blocks.2"中提取"2"
            parts = layer_name.split('.')
            if len(parts) >= 2 and parts[0] == 'blocks':
                try:
                    layer_idx = int(parts[1])
                    layer_indices.append((layer_idx, layer_name))
                except ValueError:
                    logger.warning(f"无法从层名称 {layer_name} 中提取索引")
        
        # 按索引排序层
        layer_indices.sort(key=lambda x: x[0])
        
        if not layer_indices:
            logger.warning("未找到有效的层索引，Propagate模式无法应用")
            return
            
        # 获取第一层的名称
        first_layer_idx, first_layer_name = layer_indices[0]
        
        # 为每个层设置步骤
        for i, (layer_idx, layer_name) in enumerate(layer_indices):
            # 自注意力块和交叉注意力块
            attn_block_key = f"{layer_name}_attn_block"
            cross_attn_block_key = f"{layer_name}_cross_attn_block"
            ffn_block_key = f"{layer_name}_ffn_block"
            
            # 所有层的注意力块都使用所有步骤计算
            block_steps[attn_block_key] = []
            block_steps[cross_attn_block_key] = []
            
            # 第一层的FFN块只在step 0计算，其他步骤使用缓存
            if i == 0:  # 第一层
                # 只在步骤0计算，其他使用缓存
                block_steps[ffn_block_key] = [0]
                logger.info(f"第一个FFN块 {ffn_block_key} 只在step 0计算，其他步骤使用缓存")
            else:  # 后续层
                # 所有步骤都计算
                block_steps[ffn_block_key] = []
                logger.info(f"后续FFN块 {ffn_block_key} 在所有步骤都计算")
        
        # 存储到缓存
        cache['block_steps'] = block_steps
        
        # 日志输出
        logger.info(f"Propagate模式：({first_layer_name}_ffn_block) 只在step 0计算，其他步骤使用缓存；后续FFN Block全部计算")

    @staticmethod
    def _setup_edit_mode(cache: Dict, layers: List[Tuple[str, Any]]):
        """
        Edit模式：所有block共用一个通过参数指定的steps
        
        Args:
            cache: 缓存字典
            layers: 层列表 [(name, layer), ...]
        """
        # 获取自定义steps
        edit_steps = cache.get('edit_steps', [])
        
        # 如果没有提供自定义steps，使用默认的step 0
        if not edit_steps:
            logger.warning("未提供自定义steps，将使用默认步骤[0]")
            edit_steps = [0]
        
        # 存储每个块的步骤
        block_steps = {}
        
        # 为所有层的所有块分配相同的steps
        for layer_name, _ in layers:
            # 自注意力块
            attn_block_key = f"{layer_name}_attn_block"
            block_steps[attn_block_key] = edit_steps.copy()
            
            # 交叉注意力块
            cross_attn_block_key = f"{layer_name}_cross_attn_block"
            block_steps[cross_attn_block_key] = edit_steps.copy()
            
            # 前馈网络块
            ffn_block_key = f"{layer_name}_ffn_block"
            block_steps[ffn_block_key] = edit_steps.copy()
            
            logger.info(f"为层 {layer_name} 的所有块分配自定义steps: {edit_steps}")
        
        # 存储到缓存
        cache['block_steps'] = block_steps
        
        # 日志输出
        logger.info(f"Edit模式：已为所有block分配自定义steps: {edit_steps}")

    @staticmethod 
    def _update_cache_flags(cache: Dict, current_step: int):
        """
        更新所有块的缓存标志
        
        Args:
            cache: 缓存字典
            current_step: 当前步骤
        """
        cache_mode = cache['mode']

        if cache_mode == 'threshold':
            # 基于阈值的缓存策略
            threshold = cache['threshold']
            
            # 每隔threshold步重新计算一次
            should_compute = (current_step % threshold == 0)
            
            cache['should_compute'] = should_compute

        else:
            # 基于最优步骤、Fix模式、Propagate模式或Edit模式的缓存策略
            # 性能优化：使用字典预先存储所有steps，然后进行O(1)查找
            if 'steps_lookup' not in cache:
                # 首次运行时创建查找表
                steps_lookup = {}
                for block_key, steps in cache['block_steps'].items():
                    # 空步骤列表表示所有步骤都计算（不缓存）
                    if not steps:
                        steps_lookup[block_key] = set()
                    else:
                        steps_lookup[block_key] = set(steps)
                cache['steps_lookup'] = steps_lookup

            # 使用查找表进行O(1)查找
            # optimal steps表示"需要重新计算的步骤"，不在这些步骤中的可以使用缓存
            should_compute = {}
            for block_key, step_set in cache['steps_lookup'].items():
                # 空步骤集合表示所有步骤都计算（不缓存）
                if not step_set:
                    should_compute[block_key] = True  # 所有步骤都计算
                else:
                    should_compute[block_key] = current_step in step_set  # 在optimal steps中需要计算

            cache['should_compute'] = should_compute 

    @staticmethod
    def _add_cache_to_block(layer, layer_name: str, cache: Dict):
        """
        为RDT层添加缓存功能，缓存主要计算块的结果
        
        Args:
            layer: RDT层 (RDTBlock)
            layer_name: 层名称
            cache: 缓存字典
        """
        # 保存原始forward方法
        original_forward = layer.forward

        # 获取是否启用插值
        interpolation_ratio = cache.get('interpolation_ratio', 1.0)
        reference_activations = cache.get('reference_activations', None)

        # 辅助函数：对缓存激活与参考激活进行插值
        def interpolate_activations(cache_act, reference_activations, block_key, timestep, interpolation_ratio):
            """在缓存的激活值和参考激活值之间进行线性插值"""
            # 检查缓存激活值是否为None
            if cache_act is None:
                return None
                
            if reference_activations is None:
                return cache_act
            
            # 获取 cache_act 的设备
            device = cache_act.device
            
            # 定义块与dropout的映射关系
            ref_key = None
            
            # 提取layer_name部分 (如 blocks.7)
            if "_attn_block" in block_key:
                layer_name = block_key.replace("_attn_block", "")
                ref_key = f"{layer_name}.attn.dropout"
            elif "_cross_attn_block" in block_key:
                layer_name = block_key.replace("_cross_attn_block", "")
                ref_key = f"{layer_name}.cross_attn.dropout"
            elif "_ffn_block" in block_key:
                layer_name = block_key.replace("_ffn_block", "")
                ref_key = f"{layer_name}.ffn.dropout"
            
            if ref_key and ref_key in reference_activations and timestep < len(reference_activations[ref_key]):
                ref_act = reference_activations[ref_key][timestep]
                # 确保 ref_act 在正确的设备上
                ref_act = ref_act.to(device)
                
                if ref_act.shape == cache_act.shape:
                    return interpolation_ratio * cache_act + (1 - interpolation_ratio) * ref_act
            
            return cache_act  # 如果找不到匹配的参考激活值，返回原始缓存激活值

        # 为RDT层创建缓存forward方法
        def forward_with_cache(self, x, c, mask=None):
            """
            RDT Block的缓存forward方法
            
            Args:
                x: input tensor
                c: condition tensor  
                mask: attention mask
            """
            cache_mode = cache['mode']
            current_step = cache.get('current_step', 0)
            
            # 缓存键
            attn_block_key = f"{layer_name}_attn_block"
            cross_attn_block_key = f"{layer_name}_cross_attn_block" 
            ffn_block_key = f"{layer_name}_ffn_block"

            # 确定是否应该重新计算
            if cache_mode == 'threshold':
                should_compute = cache.get('should_compute', True)
                should_compute_attn = should_compute
                should_compute_cross_attn = should_compute
                should_compute_ffn = should_compute
            else:
                block_should_compute = cache.get('should_compute', {})
               
                should_compute_attn = block_should_compute.get(attn_block_key, True)
                should_compute_cross_attn = block_should_compute.get(cross_attn_block_key, True)
                should_compute_ffn = block_should_compute.get(ffn_block_key, True)

            # 检查是否可以使用缓存（不需要重新计算且有缓存）
            can_use_attn_cache = not should_compute_attn and attn_block_key in cache['block_cache']
            can_use_cross_attn_cache = not should_compute_cross_attn and cross_attn_block_key in cache['block_cache']
            can_use_ffn_cache = not should_compute_ffn and ffn_block_key in cache['block_cache']

            # 调试日志（每20步输出一次以避免日志过多）
            if current_step % 20 == 0:
                logger.debug(f"Step {current_step}, Layer {layer_name}: "
                           f"compute_attn={should_compute_attn}, compute_cross={should_compute_cross_attn}, compute_ffn={should_compute_ffn}, "
                           f"cache_attn={can_use_attn_cache}, cache_cross={can_use_cross_attn_cache}, cache_ffn={can_use_ffn_cache}")

            # 如果所有主要块都可以使用缓存，直接返回缓存结果
            if can_use_attn_cache and can_use_cross_attn_cache and can_use_ffn_cache:
                # 优先使用最终输出缓存
                final_cache_key = ffn_block_key + '_final'
                if final_cache_key in cache['block_cache']:
                    final_cached_value = cache['block_cache'][final_cache_key]
                    if final_cached_value is not None:
                        # 应用插值（如果启用）
                        final_cache = interpolate_activations(final_cached_value, reference_activations, ffn_block_key, current_step, interpolation_ratio)
                        return final_cache
                
                # 如果没有最终输出缓存，执行原始forward方法
                output = original_forward(x, c, mask)
                return output

            # 按照RDTBlock的结构逐步计算和缓存
            # 1. Self-Attention Block
            origin_x = x
            x = self.norm1(x)
            
            if can_use_attn_cache:
                # 使用缓存的self-attention结果
                attn_cached_value = cache['block_cache'][attn_block_key]
                attn_result = interpolate_activations(attn_cached_value, reference_activations, attn_block_key, current_step, interpolation_ratio)
            else:
                # 计算self-attention
                attn_result = self.attn(x)
                if should_compute_attn:
                    cache['block_cache'][attn_block_key] = attn_result.detach() if isinstance(attn_result, torch.Tensor) else attn_result
            
            x = attn_result + origin_x
            
            # 2. Cross-Attention Block  
            origin_x = x
            x = self.norm2(x)
            
            if can_use_cross_attn_cache:
                # 使用缓存的cross-attention结果
                cross_attn_cached_value = cache['block_cache'][cross_attn_block_key]
                cross_attn_result = interpolate_activations(cross_attn_cached_value, reference_activations, cross_attn_block_key, current_step, interpolation_ratio)
            else:
                # 计算cross-attention
                cross_attn_result = self.cross_attn(x, c, mask)
                if should_compute_cross_attn:
                    cache['block_cache'][cross_attn_block_key] = cross_attn_result.detach() if isinstance(cross_attn_result, torch.Tensor) else cross_attn_result
            
            x = cross_attn_result + origin_x
            
            # 3. FFN Block
            origin_x = x
            x = self.norm3(x)
            
            if can_use_ffn_cache:
                # 使用缓存的FFN结果
                ffn_cached_value = cache['block_cache'][ffn_block_key]
                ffn_result = interpolate_activations(ffn_cached_value, reference_activations, ffn_block_key, current_step, interpolation_ratio)
            else:
                # 计算FFN
                ffn_result = self.ffn(x)
                if should_compute_ffn:
                    cache['block_cache'][ffn_block_key] = ffn_result.detach() if isinstance(ffn_result, torch.Tensor) else ffn_result
            
            x = ffn_result + origin_x
            
            # 缓存最终输出（用于完全缓存的情况）
            if should_compute_ffn:  # 只有在FFN重新计算时才更新最终输出
                cache['block_cache'][ffn_block_key + '_final'] = x.detach() if isinstance(x, torch.Tensor) else x

            return x

        # 替换层的forward方法
        layer.forward = types.MethodType(forward_with_cache, layer) 