# -*- coding: utf-8 -*-
"""
LightGBM优化版网格搜索（适配信用数据）
"""
import warnings
import sys
from datetime import datetime
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
import os
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid, train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import time
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._classification")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

# 初始化配置
# --------------------------------------------------
start_time = time.time()
credit = pd.read_csv("sorted_data_xgb_notime.csv", encoding='gbk')
credit2 = pd.read_csv("filtered_data_xgb_notime.csv", encoding='gbk')
credit3 = pd.read_csv("排序少样本_data_xgb.csv", encoding='gbk')

# 数据准备
# --------------------------------------------------
X = credit.iloc[0:1610, :20]
y = credit.iloc[0:1610, -1]
X_train = credit2.iloc[0:1371, :20]
y_train = credit2.iloc[0:1371, -1]
X_out = credit3.iloc[0:97, :20]
y_out = credit3.iloc[0:97, -1]

# 计算类别权重
pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)  # 自动计算正样本权重

# 划分测试集
_, X_test, _, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 重采样处理
# --------------------------------------------------
# 原始数据
resampled_datasets = []
resampled_datasets.append(('original', X_train, y_train))

# RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_train, y_train)
resampled_datasets.append(('OverSample', X_ros, y_ros))

# RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train, y_train)
resampled_datasets.append(('UnderSample', X_rus, y_rus))

# SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
resampled_datasets.append(('SMOTE', X_smote, y_smote))
kfold=StratifiedKFold(n_splits=3)

# 日志配置
# --------------------------------------------------
log_dir = "../logmy"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"lgb_1_model_{datetime.now().strftime('%Y%m%d%H%M')}.log")


class Logger:
    def __init__(self, file_path):
        self.console = sys.stdout
        self.file = open(file_path, "a", encoding="utf-8")

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


sys.stdout = Logger(log_file)


# 核心训练函数
# --------------------------------------------------
def enhanced_trainer(resampled_data, params_grid):
    method_name, X_resampled, y_resampled = resampled_data
    best_models = []
    print(f"\n{'=' * 40}\n采样方法: {method_name}\n{'=' * 40}")

    for weight in params_grid['scale_pos_weight']:
        print(f"\n当前权重: {weight}")
        print(method_name)

        for param_combo in ParameterGrid(params_grid['main_params']):
            print(f"\n当前权重: {weight}")
            print(method_name)
            print(param_combo)
            model = lgb.LGBMClassifier(
                **param_combo,
                scale_pos_weight=weight,
                random_state=42,
                verbosity=-1,
            force_row_wise= True  # 防止多线程建议输出
            )

            try:
                model.fit(X_resampled, y_resampled)
            except Exception as e:
                print(f"参数组合 {param_combo} 训练失败: {str(e)}")
                continue

            # 评估指标
            test_report = evaluate_model(model, X_test, y_test)
            val_report = evaluate_model(model, X_out, y_out)
            print()
            print(test_report)
            print(val_report)

            if validate_conditions(test_report, val_report):
                best_models.append({
                    'method': method_name,
                    'params': param_combo,
                    'weight': weight,
                    'test_metrics': test_report,
                    'val_metrics': val_report
                })

    return best_models


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    return {
        'accuracy': accuracy_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_prob),
        'recall': recall_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'confusion': confusion_matrix(y, y_pred)
    }


def validate_conditions(test, val):
    return all([
        test['accuracy'] > val['accuracy'],
        test['recall'] >= val['recall'],
        abs(test['roc_auc'] - val['roc_auc']) <= 0.1,
        test['accuracy'] >= 0.96,
        test['recall'] >= 0.80
    ])


# 参数配置（LightGBM优化版）
# --------------------------------------------------
parameters = {
    'main_params': {
        'num_leaves': [31, 63, 127,7,10,8],  # 典型设置：2^max_depth -1
        'max_depth': [5, 7, -1,30,12,10],  # -1表示不限制
        'learning_rate': [0.1,0.01,0.015,0.02,0.03,0.025],
        'n_estimators': [200, 100,30,50,80,150],
        'reg_alpha': [0, 0.1],  # L1正则
        'reg_lambda': [0, 0.1] , # L2正则

    },
    'scale_pos_weight': [
        pos_weight,  # 自动计算的平衡权重
        2.19, 2.195, 2.2, 2.185,0.456  # 实验性权重参数
    ]
}

# 执行训练
# --------------------------------------------------
final_results = []
for data_pack in resampled_datasets:
    results = enhanced_trainer(data_pack, parameters)
    final_results.extend(results)

# 结果展示优化
# --------------------------------------------------
print("\n最优模型汇总（按验证集AUC排序）：")
sorted_models = sorted(final_results,
                       key=lambda x: x['val_metrics']['roc_auc'],
                       reverse=True)[:]  # 显示前10个最佳模型

for idx, model in enumerate(sorted_models, 1):
    print(f"\n模型 {idx}:")
    print(f"采样方法: {model['method']}")
    print(f"参数: {model['params']}")
    print(f"权重: {model['weight']:.3f}")
    print(f"测试集 AUC: {model['test_metrics']['roc_auc']:.3f} 验证集 AUC: {model['val_metrics']['roc_auc']:.3f}")
    print(f"召回率: {model['test_metrics']['recall']:.3f} | 准确率: {model['test_metrics']['accuracy']:.3f}")

# 恢复输出
sys.stdout.file.close()
sys.stdout = sys.stdout.console

# 性能统计
# --------------------------------------------------
end_time = time.time()
print(f"\n总运行时间: {end_time - start_time:.2f}秒")
print(f"评估模型总数: {len(final_results)}")
print(f"达标模型数量: {len(final_results)}")
