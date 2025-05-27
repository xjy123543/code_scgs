# -*- coding: utf-8 -*-
"""
优化后的网格搜索脚本（适配信用数据）- 特征自适应版本
"""
import warnings
import sys
from datetime import datetime
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import time
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

# 初始化配置
# --------------------------------------------------
start_time = time.time()


# 数据加载校验函数
def load_data_with_validation():
    def read_csv_safe(path):
        df = pd.read_csv(path, encoding='gbk')
        assert df.shape[0] > 0, f"{path} 为空或路径错误"
        return df

    credit = read_csv_safe("sorteddataset_Decisiontree_notime.csv")
    credit2 = read_csv_safe("filtered_data_Decisiontree_notime.csv")
    credit3 = read_csv_safe("排序少样本_data.csv")

    # 自动生成特征顺序（按源数据列顺序）
    original_features = credit.columns[:20].tolist()
    print(f"自动特征排序: {original_features[:20]}")

    return credit, credit2, credit3, original_features[:20]  # 只取前20


# 加载数据
credit, credit2, credit3, FEATURE_ORDER = load_data_with_validation()


# 数据准备（统一特征空间）
# --------------------------------------------------
def safe_feature_select(df, features):
    """安全特征选择，确保列存在"""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"缺失关键特征列: {missing}")
    return df[features]


# 数据统一处理
X = safe_feature_select(credit.iloc[0:], FEATURE_ORDER)
y = credit.iloc[0:, -1]
X_train = safe_feature_select(credit2.iloc[0:], FEATURE_ORDER)
y_train = credit2.iloc[0:, -1]
X_out = safe_feature_select(credit3.iloc[0:], FEATURE_ORDER)
y_out = credit3.iloc[0:, -1]

_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42
)

# 重采样处理
# --------------------------------------------------
resampled_datasets = []
resamplers = [
    None,  # 原始数据
    RandomOverSampler(random_state=42),
    RandomUnderSampler(random_state=42),
    SMOTE(random_state=42)
]

for resampler in resamplers:
    name = resampler.__class__.__name__ if resampler else "Original"
    if resampler:
        X_res, y_res = resampler.fit_resample(X_train, y_train)
    else:
        X_res, y_res = X_train.copy(), y_train.copy()
    resampled_datasets.append((name, X_res, y_res))


# 动态特征选择框架
# --------------------------------------------------
class DynamicFeatureSelector:
    def __init__(self, base_estimator, min_features=10):
        self.selector = RFECV(
            estimator=base_estimator,
            step=1,
            cv=StratifiedKFold(10),
            min_features_to_select=14,

            scoring='accuracy'
        )

    def fit_transform(self, X, y):
        self.selector.fit(X, y)
        return X.columns[self.selector.support_].tolist()

    @property
    def n_features(self):
        return self.selector.n_features_


# 核心训练逻辑
# --------------------------------------------------
def enhanced_trainer(resampled_data, params_grid):
    method_name, X_resampled, y_resampled = resampled_data
    best_models = []

    print(f"\n{'-' * 30} {method_name} {'-' * 30}")

    for weight in params_grid['class_weight']:
        for params in ParameterGrid(params_grid['main_params']):
            try:
                # 初始化基础模型
                base_model = RandomForestClassifier(
                    class_weight=weight,
                    **params,
                    random_state=42
                )

                # 特征选择
                selector = DynamicFeatureSelector(base_model)
                selected_features = selector.fit_transform(X_resampled, y_resampled)

                # 生成对应数据集
                X_train_clean = X_resampled[selected_features]
                X_test_clean = X_test[selected_features]
                X_out_clean = X_out[selected_features]

                # 最终模型训练
                final_model = RandomForestClassifier(
                    class_weight=weight,
                    **params,
                    random_state=42
                ).fit(X_train_clean, y_resampled)

                # 评估环节
                print(weight)
                print(params)
                print(len(selected_features))
                test_metrics = evaluate_model(final_model, X_test_clean, y_test)
                val_metrics = evaluate_model(final_model, X_out_clean, y_out)
                print( test_metrics)
                print(val_metrics)


                if validate_conditions(test_metrics, val_metrics):
                    best_models.append({
                        'method': method_name,
                        'params': params,
                        'class_weight': weight,
                        'features': selected_features,
                        'test': test_metrics,
                        'val': val_metrics
                    })

            except Exception as e:
                print(f"参数组合失败: {params}|{weight} -> {str(e)}")

    return best_models


# 评估验证工具函数
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        'accuracy': round(accuracy_score(y, y_pred), 4),
        'recall': round(recall_score(y, y_pred), 4),
        'roc_auc': round(roc_auc_score(y, y_prob), 4),
        'confusion': confusion_matrix(y, y_pred)
    }


def validate_conditions(test, val):
    return (
            (test['accuracy'] >= 0.97) &
            (val['accuracy'] >= 0.9) &
            (abs(test['roc_auc'] - val['roc_auc']) <= 0.05) &
            (test['recall'] >= 0.8)
    )


# 参数配置
parameters = {
    'main_params': {
        'max_depth': [15,20, 25, None,12,],
        'n_estimators': [80,100, 200,30,],
        'max_features': [12,'sqrt', 8,10,5,3]
    },
    'class_weight': [
        {0: 1, 1: 2.19},'balanced',
{0: 1, 1: 5.7},{0: 1, 1: 5.5},{0: 1, 1: 5.4},
        {0: 1, 1: 2.18},

        {0: 1, 1: 2.195},

    ]
}


# 日志配置（保持原有）
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


if __name__ == "__main__":
    # 运行逻辑
    log_dir = "../logmy"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"auto_feature_rf_1_{datetime.now().strftime('%Y%m%d%H%M')}.log")
    sys.stdout = Logger(log_file)

    final_results = []
    for data_pack in resampled_datasets:
        results = enhanced_trainer(data_pack, parameters)
        final_results.extend(results)

    # 结果展示优化
    columns = ["Method", "Features", "Test AUC", "Val AUC", "Params"]
    summary = pd.DataFrame([{
        "Method": r['method'],
        "Features": len(r['features']),
        "Test AUC": r['test']['roc_auc'],
        "Val AUC": r['val']['roc_auc'],
        "Params": str(r['params'])
    } for r in final_results])

    print("\n综合结果汇总:")
    print(summary.sort_values("Val AUC", ascending=False).head(10))

    sys.stdout.file.close()
    sys.stdout = sys.stdout.console
    print(f"总耗时: {time.time() - start_time:.1f}s")
