#!/bin/bash

# ==============================================================================
# 综合实验运行脚本
# 包含 CTransPath (40x, 100x, Mixed) 和 DAF-MMD (Xception-Swin, Xception-Xception)
# ==============================================================================

# 激活虚拟环境
source venv_cancer/bin/activate

# 错误处理函数
handle_error() {
    echo "❌ 实验失败: $1"
    # 不退出，继续下一个实验
}

echo "================================================================================"
echo "开始运行所有实验"
echo "================================================================================"

# ------------------------------------------------------------------------------
# 1. CTransPath 实验 (40x, 100x, Mixed)
# ------------------------------------------------------------------------------
echo ""
echo "################################################################################"
echo "阶段 1: CTransPath 实验"
echo "################################################################################"

CTRANSPATH_CONFIGS=(
    "configs/multitask_ctranspath_40x.yaml"
    "configs/multitask_ctranspath_100x.yaml"
    "configs/multitask_ctranspath_mixed.yaml"
)

for CONFIG in "${CTRANSPATH_CONFIGS[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "正在训练: $CONFIG"
    echo "--------------------------------------------------------------------------------"
    
    # 提取实验名称用于显示
    EXP_NAME=$(grep "experiment_name:" $CONFIG | awk -F'"' '{print $2}')
    if [ -z "$EXP_NAME" ]; then
        EXP_NAME=$(grep "experiment_name:" $CONFIG | awk '{print $2}')
    fi
    
    # 训练
    python train_multitask.py --config $CONFIG
    
    if [ $? -eq 0 ]; then
        # 评估
        echo "正在评估..."
        CHECKPOINT_DIR="multitask_results/checkpoints/$EXP_NAME"
        if [ -d "$CHECKPOINT_DIR" ]; then
            python evaluate_multitask.py --checkpoint_dir $CHECKPOINT_DIR --config $CONFIG
        else
            echo "⚠️  未找到 Checkpoint 目录: $CHECKPOINT_DIR"
        fi
        echo "✓ $EXP_NAME 完成"
    else
        handle_error "$EXP_NAME"
    fi
done

# ------------------------------------------------------------------------------
# 2. DAF-MMD 实验 (Xception-Swin, Xception-Xception)
# ------------------------------------------------------------------------------
echo ""
echo "################################################################################"
echo "阶段 2: DAF-MMD 实验"
echo "################################################################################"

DAF_CONFIGS=(
    "configs/daf_mmd_xception_swin.yaml"
    "configs/daf_mmd_xception_xception.yaml"
)

for CONFIG in "${DAF_CONFIGS[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "正在训练: $CONFIG"
    echo "--------------------------------------------------------------------------------"
    
    # 提取实验名称用于目录查找 (通常 config 文件名对应目录名，或者在 yaml 中定义)
    # 这里我们直接运行，并在成功后评估
    
    python train_daf_mmd_xception.py --config $CONFIG
    
    if [ $? -eq 0 ]; then
        # 评估
        echo "正在评估..."
        
        # 根据 config 文件名推断 checkpoint 目录名 (这是一个假设，需根据实际 yaml 确认)
        if [[ $CONFIG == *"swin"* ]]; then
            CHECKPOINT_DIR="multitask_results/checkpoints/daf_mmd_xception_swin"
        else
            CHECKPOINT_DIR="multitask_results/checkpoints/daf_mmd_xception_xception"
        fi
        
        if [ -d "$CHECKPOINT_DIR" ]; then
            python evaluate_daf_mmd_xception.py --checkpoint_dir $CHECKPOINT_DIR
        else
             # 尝试从 yaml 中读取 (简单的 grep)
            SAVE_DIR=$(grep "save_dir:" $CONFIG | head -n 1 | awk '{print $2}' | tr -d '"')
            if [ -d "$SAVE_DIR" ]; then
                 python evaluate_daf_mmd_xception.py --checkpoint_dir $SAVE_DIR
            else
                echo "⚠️  未找到 Checkpoint 目录，请手动检查评估"
            fi
        fi
        echo "✓ DAF-MMD ($CONFIG) 完成"
    else
        handle_error "DAF-MMD ($CONFIG)"
    fi
done

echo ""
echo "================================================================================"
echo "所有实验流程结束"
echo "================================================================================"

