#!/bin/bash

# RDT模型获取所有任务的optimal steps脚本
# 此脚本将为RDT的ManiSkill任务计算最优缓存步骤

# 默认参数
DEVICE="cuda:0"
NUM_CACHES="5,10"
METRICS="cosine,mse,l1"
FORCE_RECOMPUTE="--force_recompute"
DEBUG_MODE=false  # 调试模式参数

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num_caches)
            NUM_CACHES="$2"
            shift 2
            ;;
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        --no_force)
            FORCE_RECOMPUTE=""
            shift
            ;;
        --task)
            SINGLE_TASK="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 定义RDT ManiSkill任务列表及其检查点
declare -A RDT_TASKS=(
    ["PickCube-v1"]="/home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt"
    ["StackCube-v1"]="/home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt"
    ["PegInsertionSide-v1"]="/home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt"
    ["PlugCharger-v1"]="/home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt"
    ["PushCube-v1"]="/home/kyji/public/public_data/models/maniskill-model/rdt/mp_rank_00_model_states.pt"
)

# 处理单个任务的情况
if [ ! -z "$SINGLE_TASK" ]; then
    if [[ -v RDT_TASKS["$SINGLE_TASK"] ]]; then
        echo "仅处理RDT任务: $SINGLE_TASK"
        # 创建一个新的关联数组，只包含指定任务
        declare -A RDT_TASKS_FILTERED
        RDT_TASKS_FILTERED["$SINGLE_TASK"]="${RDT_TASKS[$SINGLE_TASK]}"
        RDT_TASKS=()
        # 正确方式是复制每一个键值对
        for key in "${!RDT_TASKS_FILTERED[@]}"; do
            RDT_TASKS["$key"]="${RDT_TASKS_FILTERED[$key]}"
        done
    else
        echo "错误: 找不到任务 $SINGLE_TASK"
        echo "可用任务: ${!RDT_TASKS[@]}"
        exit 1
    fi
fi

# 显示任务信息
echo "===========================================" 
echo "开始获取RDT最优缓存步骤"
echo "===========================================" 
echo "设备: $DEVICE"
echo "缓存数量: $NUM_CACHES"
echo "相似度指标: $METRICS"
if [ "$DEBUG_MODE" = true ]; then
    echo "调试模式: 开启 (只打印任务信息，不执行命令)"
fi
if [ ! -z "$FORCE_RECOMPUTE" ]; then
    echo "强制重新计算: 是"
else
    echo "强制重新计算: 否"
fi
echo "任务数量: ${#RDT_TASKS[@]}"
echo "任务列表: ${!RDT_TASKS[@]}"
echo "===========================================" 

# 创建assets目录
mkdir -p assets

# 处理RDT任务
if [ ${#RDT_TASKS[@]} -gt 0 ]; then
    echo "===========================================" 
    echo "处理RDT ManiSkill任务..."
    echo "===========================================" 
    
    for task_name in "${!RDT_TASKS[@]}"; do
        checkpoint="${RDT_TASKS[$task_name]}"
        if [ ! -f "$checkpoint" ]; then
            echo "警告: 检查点文件不存在: $checkpoint, 跳过任务 $task_name"
            continue
        fi

        output_dir="assets/${task_name}"
        mkdir -p "$output_dir"
        
        echo "处理RDT任务: $task_name"
        echo "检查点: $checkpoint"
        echo "输出目录: $output_dir"
        
        # 调试模式下跳过执行
        if [ "$DEBUG_MODE" = true ]; then
            echo "调试模式: 跳过执行"
            continue
        fi
        
        # 运行命令
        cmd="python -m models.activation_utils.get_optimal_cache_update_steps_rdt -c $checkpoint -o $output_dir -d $DEVICE --num_caches \"${NUM_CACHES}\" --metrics \"${METRICS}\" ${FORCE_RECOMPUTE}"
        echo "执行命令: $cmd"
        eval $cmd
        
        if [ $? -eq 0 ]; then
            echo "任务 ${task_name} 完成"
        else
            echo "任务 ${task_name} 失败"
        fi
        echo "-------------------------------------------"
    done
fi

echo "===========================================" 
echo "所有RDT任务处理完成"
echo "===========================================" 

# 生成总结报告
summary_file="assets/rdt_optimal_steps_summary.txt"
echo "生成总结报告: $summary_file"

cat > "$summary_file" << EOF
RDT模型最优缓存步骤总结报告
========================================

生成时间: $(date)
处理设备: $DEVICE
缓存数量: $NUM_CACHES
相似度指标: $METRICS
强制重新计算: $(if [ ! -z "$FORCE_RECOMPUTE" ]; then echo "是"; else echo "否"; fi)

处理的任务:
EOF

for task_name in "${!RDT_TASKS[@]}"; do
    checkpoint="${RDT_TASKS[$task_name]}"
    echo "- $task_name: $checkpoint" >> "$summary_file"
done

echo "" >> "$summary_file"
echo "输出文件结构:" >> "$summary_file"
echo "assets/" >> "$summary_file"
echo "├── TaskName-v1/" >> "$summary_file"
echo "│   ├── original/" >> "$summary_file"
echo "│   │   ├── activations.pkl" >> "$summary_file"
echo "│   │   └── similarity_matrices.pkl" >> "$summary_file"
echo "│   ├── optimal_steps_module_name_N_metric.pkl" >> "$summary_file"
echo "│   ├── all_optimal_steps.pkl" >> "$summary_file"
echo "│   └── optimal_steps_summary.txt" >> "$summary_file"
echo "└── rdt_optimal_steps_summary.txt (此文件)" >> "$summary_file"

echo "总结报告已生成: $summary_file"

# 使用说明
usage() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  --device device_name        指定要使用的设备 (默认: cuda:0)"
    echo "  --num_caches list           指定要计算的缓存数量，逗号分隔 (默认: 5,10,20,30)"
    echo "  --metrics list              指定要计算的相似度指标，逗号分隔 (默认: cosine,mse,l1)"
    echo "  --no_force                  不强制重新计算"
    echo "  --task task_name            只处理指定的任务"
    echo "  --debug                     调试模式，只打印任务信息，不执行命令"
    echo
    echo "可用任务:"
    for task in "${!RDT_TASKS[@]}"; do
        echo "  - $task"
    done
    echo
    echo "示例:"
    echo "  $0 --device cuda:1 --num_caches 5,10,15 --metrics cosine --task PickCube-v1"
    echo "  $0 --device cuda:0 --num_caches 5,10,15,20,30"
    echo "  $0 --task PickCube-v1 --debug"
}

# 如果没有参数则显示使用说明
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    usage
    exit 0
fi 