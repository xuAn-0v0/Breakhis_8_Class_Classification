#!/bin/bash

# DAF-MMD with Xception 实验运行脚本
# 运行xception_swin和xception_xception两种配置，以及multitask的xception

echo "================================================================================"
echo "开始运行 DAF-MMD with Xception 实验"
echo "================================================================================"

# 激活虚拟环境
source venv_cancer/bin/activate

# 1. 运行 DAF-MMD Xception-Swin
echo ""
echo "================================================================================"
echo "1. 训练 DAF-MMD Xception-Swin (40X-Xception + 100X-Swin)"
echo "================================================================================"
python train_daf_mmd_xception.py --config configs/daf_mmd_xception_swin.yaml

if [ $? -eq 0 ]; then
    echo ""
    echo "评估 DAF-MMD Xception-Swin..."
    python evaluate_daf_mmd_xception.py --checkpoint_dir multitask_results/checkpoints/daf_mmd_xception_swin
else
    echo "❌ DAF-MMD Xception-Swin 训练失败"
fi

# 2. 运行 DAF-MMD Xception-Xception
echo ""
echo "================================================================================"
echo "2. 训练 DAF-MMD Xception-Xception (40X-Xception + 100X-Xception)"
echo "================================================================================"
python train_daf_mmd_xception.py --config configs/daf_mmd_xception_xception.yaml

if [ $? -eq 0 ]; then
    echo ""
    echo "评估 DAF-MMD Xception-Xception..."
    python evaluate_daf_mmd_xception.py --checkpoint_dir multitask_results/checkpoints/daf_mmd_xception_xception
else
    echo "❌ DAF-MMD Xception-Xception 训练失败"
fi

# 3. 运行 Multitask Xception (已有的单任务Xception)
echo ""
echo "================================================================================"
echo "3. 训练 Multitask Xception (40x, 100x, mixed)"
echo "================================================================================"

CONFIGS=(
    "configs/multitask_xception_40x.yaml"
    "configs/multitask_xception_100x.yaml"
    "configs/multitask_xception_mixed.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "正在训练: $CONFIG"
    echo "--------------------------------------------------------------------------------"
    
    EXP_NAME=$(grep "experiment_name:" $CONFIG | awk -F'"' '{print $2}')
    if [ -z "$EXP_NAME" ]; then
        EXP_NAME=$(grep "experiment_name:" $CONFIG | awk '{print $2}')
    fi
    
    # 训练
    python train_multitask.py --config $CONFIG
    
    if [ $? -eq 0 ]; then
        # 评估
        CHECKPOINT_DIR="multitask_results/checkpoints/$EXP_NAME"
        if [ -d "$CHECKPOINT_DIR" ]; then
            python evaluate_multitask.py --checkpoint_dir $CHECKPOINT_DIR --config $CONFIG
        fi
        echo "✓ $EXP_NAME 完成"
    else
        echo "❌ $EXP_NAME 训练失败"
    fi
done

echo ""
echo "================================================================================"
echo "所有实验完成！"
echo "================================================================================"
echo ""
echo "结果目录："
echo "  - DAF-MMD Xception-Swin: multitask_results/results/daf_mmd_xception_swin"
echo "  - DAF-MMD Xception-Xception: multitask_results/results/daf_mmd_xception_xception"
echo "  - Multitask Xception: multitask_results/results/multitask_xception_*"
echo ""




