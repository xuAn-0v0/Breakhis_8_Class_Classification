#!/bin/bash

# 遇到错误立即停止
# set -e

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "================================================================================"
echo "开始运行剩余的评估步骤 (如果模型存在)"
echo "时间: $(date)"
echo "================================================================================"

# 定义评估函数
evaluate_if_exists() {
    MODEL_DIR=$1
    MODEL_TYPE=$2  # 'multitask' or 'daf_mmd'
    CONFIG_NAME=$3
    
    echo "--------------------------------------------------------------------------------"
    echo "检查模型: $MODEL_DIR"
    
    if [ -f "$MODEL_DIR/best_model.pth" ] || [ -f "$MODEL_DIR/last_model.pth" ]; then
        echo "✓ 发现模型文件，开始评估..."
        if [ "$MODEL_TYPE" == "multitask" ]; then
             python evaluate_multitask.py --checkpoint_dir "$MODEL_DIR"
        elif [ "$MODEL_TYPE" == "daf_mmd" ]; then
             python evaluate_daf_mmd_xception.py --checkpoint_dir "$MODEL_DIR"
        fi
    else
        echo "⚠️  未找到模型文件 (best_model.pth 或 last_model.pth)"
        echo "⚠️  将尝试重新训练..."
        
        if [ "$MODEL_TYPE" == "multitask" ]; then
             python train_multitask.py --config "configs/$CONFIG_NAME"
             # 训练后再次尝试评估
             if [ -f "$MODEL_DIR/best_model.pth" ]; then
                 python evaluate_multitask.py --checkpoint_dir "$MODEL_DIR"
             fi
        elif [ "$MODEL_TYPE" == "daf_mmd" ]; then
             python train_daf_mmd_xception.py --config "configs/$CONFIG_NAME"
             # 训练后再次尝试评估
             if [ -f "$MODEL_DIR/best_model.pth" ] || [ -f "$MODEL_DIR/last_model.pth" ]; then
                 python evaluate_daf_mmd_xception.py --checkpoint_dir "$MODEL_DIR"
             fi
        fi
    fi
    echo "--------------------------------------------------------------------------------"
}

# 1. CTransPath Multitask 实验
evaluate_if_exists "multitask_results/results/multitask_ctranspath_40x" "multitask" "multitask_ctranspath_40x.yaml"
evaluate_if_exists "multitask_results/results/multitask_ctranspath_100x" "multitask" "multitask_ctranspath_100x.yaml"
evaluate_if_exists "multitask_results/results/multitask_ctranspath_mixed" "multitask" "multitask_ctranspath_mixed.yaml"

# 2. DAF-MMD 实验
evaluate_if_exists "multitask_results/results/daf_mmd_xception_swin" "daf_mmd" "daf_mmd_xception_swin.yaml"
evaluate_if_exists "multitask_results/results/daf_mmd_xception_xception" "daf_mmd" "daf_mmd_xception_xception.yaml"

echo ""
echo "================================================================================"
echo "所有流程结束！"
echo "================================================================================"


