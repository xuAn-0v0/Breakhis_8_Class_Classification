import os
import subprocess
import yaml
import json
import pandas as pd
import time
from datetime import datetime

# 基础路径
BASE_DIR = "/songjian/zixuan"
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
RESULTS_ROOT = os.path.join(BASE_DIR, "multitask_results")
os.makedirs(CONFIG_DIR, exist_ok=True)

# 实验配置定义
BACKBONES = {
    "resnet50": {"name": "timm_multitask", "backbone": "resnet50"},
    "xception": {"name": "timm_multitask", "backbone": "xception", "image_size": 299, "batch_size": 16},
    "swin_trans": {"name": "timm_multitask", "backbone": "swin_tiny_patch4_window7_224"},
    "ctranspath": {"name": "ctranspath_multitask", "pretrained_path": "ctranspath.pth"},
    "cnn": {"name": "simple_cnn_multitask"}
}

DATA_SPLITS = {
    "40x": {
        "train_csv": "train_split_40x.csv",
        "val_csv": "val_split_40x.csv",
        "test_csv_40x": "test_split_40x.csv",
        "test_csv_100x": "test_split_100x.csv"
    },
    "100x": {
        "train_csv": "train_split_100x.csv",
        "val_csv": "val_split_100x.csv",
        "test_csv_40x": "test_split_40x.csv",
        "test_csv_100x": "test_split_100x.csv"
    },
    "mixed": {
        "train_csv": "train_split_40x_100x_mixed.csv",
        "val_csv": "val_split_40x_100x_mixed.csv",
        "test_csv_40x": "test_split_40x.csv",
        "test_csv_100x": "test_split_100x.csv"
    }
}

def create_config(backbone_key, data_key):
    backbone_info = BACKBONES[backbone_key]
    data_info = DATA_SPLITS[data_key]
    
    exp_name = f"multitask_{backbone_key}_{data_key}"
    
    config = {
        "experiment_name": exp_name,
        "group": backbone_key,
        "seed": 42,
        "device": "cuda",
        "model": {
            "name": backbone_info["name"],
            "num_classes_binary": 2,
            "num_classes_multiclass": 8,
            "use_multiclass": True,
            "dropout": 0.5
        },
        "data": {
            "dataset_root": "/songjian/zixuan/dataset_cancer_v1",
            "train_csv": data_info["train_csv"],
            "val_csv": data_info["val_csv"],
            "test_csv_40x": data_info["test_csv_40x"],
            "test_csv_100x": data_info["test_csv_100x"],
            "image_size": backbone_info.get("image_size", 224),
            "batch_size": backbone_info.get("batch_size", 32),
            "num_workers": 4
        },
        "training": {
            "epochs": 100,
            "optimizer": "adam",
            "learning_rate": 0.0001 if backbone_key != "cnn" else 0.001,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "loss": "focal_loss",
            "focal_alpha": 1.0,
            "focal_gamma": 2.0,
            "alpha_multiclass": 2.0,  # 增加八分类任务权重，提升复杂任务表现
            "early_stopping": True,
            "patience": 20            # 稍微增加耐心值，给复杂任务更多收敛空间
        },
        "checkpoint": {
            "save_dir": f"multitask_results/checkpoints/{exp_name}",
            "save_best_only": True,
            "save_freq": 10
        },
        "results": {
            "save_dir": f"multitask_results/results/{exp_name}",
            "save_predictions": True,
            "plot_confusion_matrix": True,
            "plot_training_history": True
        }
    }
    
    # 填补模型特定字段
    if "backbone" in backbone_info:
        config["model"]["backbone"] = backbone_info["backbone"]
    if "pretrained_path" in backbone_info:
        config["model"]["pretrained_path"] = backbone_info["pretrained_path"]
        
    config_path = os.path.join(CONFIG_DIR, f"{exp_name}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path, exp_name

def run_command(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def collect_results():
    all_summary = []
    
    for backbone_key in BACKBONES:
        for data_key in DATA_SPLITS:
            exp_name = f"multitask_{backbone_key}_{data_key}"
            results_dir = os.path.join(RESULTS_ROOT, "results", exp_name)
            
            # 根据实验类型选择测试子目录 (40x上训练则主要看40x测试)
            # 用户要求: 40训练40测试, 100训练100测试, mixed训练mixed测试
            test_subdir = data_key
            if data_key == "mixed":
                # Mixed 训练通常在两个测试集上评估，这里我们可以取平均或者两个都记
                # 为了简化，我们分别读取 40x 和 100x 的结果
                test_subdirs = ["40x", "100x"]
            else:
                test_subdirs = [data_key]
                
            history_path = os.path.join(results_dir, "training_history.json")
            train_time = "N/A"
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    hist = json.load(f)
                    train_time = hist.get("total_training_time_seconds", 0) / 60.0 # 分钟
            
            for sd in test_subdirs:
                metrics_path = os.path.join(results_dir, sd, "test_metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        m = json.load(f)
                        
                        row = {
                            "Model": backbone_key,
                            "Train_Data": data_key,
                            "Test_Data": sd,
                            "Train_Time_Min": f"{train_time:.2f}" if isinstance(train_time, float) else train_time,
                            "Bin_Acc": f"{m['binary']['accuracy']:.2f}%",
                            "Bin_F1_Macro": f"{m['binary']['f1_macro']:.2f}%",
                            "Bin_F1_Weighted": f"{m['binary']['f1_weighted']:.2f}%",
                            "Multi_Acc": f"{m['multiclass']['accuracy']:.2f}%" if m['multiclass'] else "N/A",
                            "Multi_F1_Macro": f"{m['multiclass']['f1_macro']:.2f}%" if m['multiclass'] else "N/A",
                            "Multi_F1_Weighted": f"{m['multiclass']['f1_weighted']:.2f}%" if m['multiclass'] else "N/A",
                        }
                        all_summary.append(row)
    
    if all_summary:
        df = pd.DataFrame(all_summary)
        summary_csv = os.path.join(RESULTS_ROOT, f"final_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(summary_csv, index=False)
        print(f"\nSummary saved to {summary_csv}")
        print(df.to_string(index=False))

def main():
    for backbone_key in BACKBONES:
        for data_key in DATA_SPLITS:
            print(f"\n>>> Starting Experiment: {backbone_key} on {data_key} <<<")
            config_path, exp_name = create_config(backbone_key, data_key)
            
            # 1. 训练
            try:
                run_command(["python", "train_multitask.py", "--config", config_path])
                
                # 2. 评估
                checkpoint_dir = os.path.join(RESULTS_ROOT, "checkpoints", exp_name)
                run_command(["python", "evaluate_multitask.py", "--checkpoint_dir", checkpoint_dir])
            except Exception as e:
                print(f"Error in experiment {exp_name}: {e}")
                continue

    # 3. 汇总结果
    collect_results()

if __name__ == "__main__":
    main()

