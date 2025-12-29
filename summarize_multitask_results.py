import os
import json
import pandas as pd
import yaml

def get_metric(data, task, metric_name):
    try:
        if task == 'binary':
            return data['binary'][metric_name]
        else:
            return data['multiclass'][metric_name]
    except (KeyError, TypeError):
        return None

def main():
    results_base_dir = "multitask_results/results"
    summary_data = []

    if not os.path.exists(results_base_dir):
        print(f"Error: 结果目录 {results_base_dir} 不存在。")
        return

    for exp_dir in os.listdir(results_base_dir):
        full_exp_path = os.path.join(results_base_dir, exp_dir)
        if not os.path.isdir(full_exp_path):
            continue

        # 加载配置
        config_path = os.path.join(full_exp_path, 'config.yaml')
        if not os.path.exists(config_path):
            continue
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 确定训练数据集类型 (40x, 100x, mixed)
        train_csv = config['data'].get('train_csv', '')
        if '40x' in train_csv and '100x' not in train_csv:
            train_type = '40x'
        elif '100x' in train_csv and '40x' not in train_csv:
            train_type = '100x'
        else:
            train_type = 'mixed'

        # 加载训练时间
        history_path = os.path.join(full_exp_path, 'training_history.json')
        train_time = "N/A"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
                seconds = history.get('total_training_time_seconds', 0)
                if seconds:
                    train_time = f"{seconds/60:.2f} min"

        # 检查测试结果
        # 用户要求: 40x训练-40x测试, 100x训练-100x测试, mixed训练-mixed测试
        # 我们总是评估 40x, 100x, 和 (逻辑上的) mixed
        
        test_types = ['40x', '100x']
        for test_type in test_types:
            metrics_path = os.path.join(full_exp_path, test_type, 'test_metrics.json')
            if not os.path.exists(metrics_path):
                continue
                
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # 记录数据
            row = {
                'Model': config['model'].get('name', exp_dir),
                'Backbone': config['model'].get('backbone', 'N/A'),
                'Train Data': train_type,
                'Test Data': test_type,
                'Binary Acc': f"{get_metric(metrics, 'binary', 'accuracy'):.2f}%",
                '8-class Acc': f"{get_metric(metrics, 'multiclass', 'accuracy'):.2f}%",
                'Binary F1 (Weighted)': f"{get_metric(metrics, 'binary', 'f1_weighted'):.2f}%",
                '8-class F1 (Weighted)': f"{get_metric(metrics, 'multiclass', 'f1_weighted'):.2f}%",
                '8-class F1 (Macro)': f"{get_metric(metrics, 'multiclass', 'f1_macro'):.2f}%",
                'Train Time': train_time
            }
            summary_data.append(row)

        # 特殊处理 Mixed 测试结果 (计算 40x 和 100x 的加权平均或从 mixed 目录读取)
        mixed_metrics_path = os.path.join(full_exp_path, 'mixed', 'test_metrics.json')
        if os.path.exists(mixed_metrics_path):
            with open(mixed_metrics_path, 'r') as f:
                metrics = json.load(f)
            row = {
                'Model': config['model'].get('name', exp_dir),
                'Backbone': config['model'].get('backbone', 'N/A'),
                'Train Data': train_type,
                'Test Data': 'mixed',
                'Binary Acc': f"{get_metric(metrics, 'binary', 'accuracy'):.2f}%",
                '8-class Acc': f"{get_metric(metrics, 'multiclass', 'accuracy'):.2f}%",
                'Binary F1 (Weighted)': f"{get_metric(metrics, 'binary', 'f1_weighted'):.2f}%",
                '8-class F1 (Weighted)': f"{get_metric(metrics, 'multiclass', 'f1_weighted'):.2f}%",
                '8-class F1 (Macro)': f"{get_metric(metrics, 'multiclass', 'f1_macro'):.2f}%",
                'Train Time': train_time
            }
            summary_data.append(row)

    if summary_data:
        df = pd.DataFrame(summary_data)
        # 按照用户要求的特定组合过滤
        # 只保留 40-40, 100-100, mixed-mixed
        filtered_df = df[((df['Train Data'] == '40x') & (df['Test Data'] == '40x')) |
                         ((df['Train Data'] == '100x') & (df['Test Data'] == '100x')) |
                         ((df['Train Data'] == 'mixed') & (df['Test Data'] == 'mixed'))]
        
        output_path = "multitask_results/final_summary.csv"
        filtered_df.to_csv(output_path, index=False)
        print(f"✓ 汇总报告已生成: {output_path}")
        print("\n部分结果预览 (40-40, 100-100, mixed-mixed):")
        print(filtered_df.to_string(index=False))
        
        # 同时保存一个包含所有评估组合的版本
        df.to_csv("multitask_results/full_summary.csv", index=False)
    else:
        print("未找到任何测试结果。")

if __name__ == "__main__":
    main()

