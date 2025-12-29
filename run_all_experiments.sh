#!/bin/bash

# 激活虚拟环境
source venv_cancer/bin/activate

# 创建配置列表
CONFIGS=(
    "configs/multitask_cnn_40x.yaml"
    "configs/multitask_cnn_100x.yaml"
    "configs/multitask_cnn_mixed.yaml"
    "configs/multitask_resnet50_40x.yaml"
    "configs/multitask_resnet50_100x.yaml"
    "configs/multitask_resnet50_mixed.yaml"
    "configs/multitask_xception_40x.yaml"
    "configs/multitask_xception_100x.yaml"
    "configs/multitask_xception_mixed.yaml"
    "configs/multitask_swin_tiny_40x.yaml"
    "configs/multitask_swin_tiny_100x.yaml"
    "configs/multitask_swin_tiny_mixed.yaml"
    "configs/multitask_ctranspath_40x.yaml"
    "configs/multitask_ctranspath_100x.yaml"
    "configs/multitask_ctranspath_mixed.yaml"
)

# 循环运行实验
for CONFIG in "${CONFIGS[@]}"; do
    echo "================================================================================"
    echo "正在开始实验: $CONFIG"
    echo "================================================================================"
    
    # 提取实验名称用于评估
    EXP_NAME=$(grep "experiment_name:" $CONFIG | awk -F'"' '{print $2}')
    if [ -z "$EXP_NAME" ]; then
        # 尝试不带引号的格式
        EXP_NAME=$(grep "experiment_name:" $CONFIG | awk '{print $2}')
    fi
    
    # 训练
    python train_multitask.py --config $CONFIG
    
    # 评估 (训练会自动在结果目录下保存 config.yaml)
    # 我们的 evaluate_multitask.py 会自动在 40x 和 100x 测试集上评估
    CHECKPOINT_DIR="multitask_results/checkpoints/$EXP_NAME"
    if [ -d "$CHECKPOINT_DIR" ]; then
        python evaluate_multitask.py --checkpoint_dir $CHECKPOINT_DIR --config $CONFIG
    else
        # 降级处理: 某些配置可能使用了不同的 save_dir 格式，从 yaml 中提取
        SAVE_DIR=$(grep "save_dir:" $CONFIG | head -n 1 | awk -F'"' '{print $2}')
        if [ -z "$SAVE_DIR" ]; then
             SAVE_DIR=$(grep "save_dir:" $CONFIG | head -n 1 | awk '{print $2}')
        fi
        python evaluate_multitask.py --checkpoint_dir $SAVE_DIR --config $CONFIG
    fi
    
    echo "实验 $EXP_NAME 完成！"
    echo ""
done

# 生成汇总报告
python summarize_multitask_results.py

echo "所有实验已完成！汇总报告见 multitask_results/final_summary.csv"

