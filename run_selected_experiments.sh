#!/bin/bash

# 遇到错误立即停止
# set -e

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "================================================================================"
echo "开始运行所有实验 (CTransPath Multitask & DAF-MMD)"
echo "时间: $(date)"
echo "================================================================================"

# ------------------------------------------------------------------------------
# 1. CTransPath Multitask Experiments (40x, 100x, Mixed)
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "1. CTransPath Multitask 实验"
echo "================================================================================"

# 1.1 CTransPath 40x
echo "--------------------------------------------------------------------------------"
echo "正在训练: CTransPath 40x"
echo "--------------------------------------------------------------------------------"
python train_multitask.py --config configs/multitask_ctranspath_40x.yaml

if [ $? -eq 0 ]; then
    echo "✓ 训练完成"
    # 获取结果目录 (假设在 multitask_results/results/multitask_ctranspath_40x)
    RESULT_DIR="multitask_results/results/multitask_ctranspath_40x"
    if [ -d "$RESULT_DIR" ]; then
        echo "正在评估..."
        python evaluate_multitask.py --checkpoint_dir "$RESULT_DIR"
    fi
else
    echo "❌ 训练失败"
fi

# 1.2 CTransPath 100x
echo "--------------------------------------------------------------------------------"
echo "正在训练: CTransPath 100x"
echo "--------------------------------------------------------------------------------"
python train_multitask.py --config configs/multitask_ctranspath_100x.yaml

if [ $? -eq 0 ]; then
    echo "✓ 训练完成"
    RESULT_DIR="multitask_results/results/multitask_ctranspath_100x"
    if [ -d "$RESULT_DIR" ]; then
        echo "正在评估..."
        python evaluate_multitask.py --checkpoint_dir "$RESULT_DIR"
    fi
else
    echo "❌ 训练失败"
fi

# 1.3 CTransPath Mixed
echo "--------------------------------------------------------------------------------"
echo "正在训练: CTransPath Mixed"
echo "--------------------------------------------------------------------------------"
python train_multitask.py --config configs/multitask_ctranspath_mixed.yaml

if [ $? -eq 0 ]; then
    echo "✓ 训练完成"
    RESULT_DIR="multitask_results/results/multitask_ctranspath_mixed"
    if [ -d "$RESULT_DIR" ]; then
        echo "正在评估..."
        python evaluate_multitask.py --checkpoint_dir "$RESULT_DIR"
    fi
else
    echo "❌ 训练失败"
fi

# ------------------------------------------------------------------------------
# 2. DAF-MMD Experiments
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo "2. DAF-MMD 实验"
echo "================================================================================"

# 2.1 DAF-MMD Xception-Swin
echo "--------------------------------------------------------------------------------"
echo "正在训练: DAF-MMD (Xception-Swin)"
echo "--------------------------------------------------------------------------------"
python train_daf_mmd_xception.py --config configs/daf_mmd_xception_swin.yaml

if [ $? -eq 0 ]; then
    echo "✓ 训练完成"
    # 获取结果目录
    RESULT_DIR="multitask_results/results/daf_mmd_xception_swin"
    if [ -d "$RESULT_DIR" ]; then
        echo "正在评估..."
        python evaluate_daf_mmd_xception.py --checkpoint_dir "$RESULT_DIR"
    fi
else
    echo "❌ 训练失败"
fi

# 2.2 DAF-MMD Xception-Xception
echo "--------------------------------------------------------------------------------"
echo "正在训练: DAF-MMD (Xception-Xception)"
echo "--------------------------------------------------------------------------------"
python train_daf_mmd_xception.py --config configs/daf_mmd_xception_xception.yaml

if [ $? -eq 0 ]; then
    echo "✓ 训练完成"
    RESULT_DIR="multitask_results/results/daf_mmd_xception_xception"
    if [ -d "$RESULT_DIR" ]; then
        echo "正在评估..."
        python evaluate_daf_mmd_xception.py --checkpoint_dir "$RESULT_DIR"
    fi
else
    echo "❌ 训练失败"
fi

echo ""
echo "================================================================================"
echo "所有实验运行结束！"
echo "================================================================================"



