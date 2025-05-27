# -*- coding: utf-8 -*-
"""
基于AdaBoost的模型评估及可视化 (max_depth=18, 15 features)
"""
import sys

sys.path.append("D:/.spyder-py3/autosave/utils")
import warnings
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier  # 修改1：添加基分类器
from sklearn.metrics import (accuracy_score, recall_score,
                             f1_score, roc_auc_score, log_loss, brier_score_loss)
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import numpy as np
import pandas as pd
from calibration_curve import plot_calibration_curve_with_ci
from dca_curve import plot_dca_with_ci
from roc_curve import plot_roc_curve_with_ci
from hl_test import hosmer_lemeshow_test

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 数据准备 --------------------------------------------------
# 修改2：调整特征列数为15
FEATURE_COUNT = 18  # 特征数常量化

# 加载数据集
credit = pd.read_csv("sorteddataset_Decisiontree_notime.csv", encoding='gbk')
credit2 = pd.read_csv("filtered_data_Decisiontree_notime.csv", encoding='gbk')
credit3 = pd.read_csv("排序少样本_data.csv", encoding='gbk')

# 特征和标签提取（修改列索引）
X = credit.iloc[0:1610, :FEATURE_COUNT]  # 使用15个特征
y = credit.iloc[0:1610, -1]
X_train = credit2.iloc[0:1371, :FEATURE_COUNT]
y_train = credit2.iloc[0:1371, -1]
X_out = credit3.iloc[0:, :FEATURE_COUNT]
y_out = credit3.iloc[0:, -1]

# 划分测试集
from sklearn.model_selection import train_test_split

_, X_test, _, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# 定义AdaBoost模型训练函数 --------------------------------------------------
def train_adaboost(params, X_train, y_train, save_path='models'):
    """训练AdaBoost模型并保存"""
    os.makedirs(save_path, exist_ok=True)

    # 修改3：配置基分类器
    base_estimator = DecisionTreeClassifier(
        max_depth=15,random_state=42
    )

    model = AdaBoostClassifier(
       estimator=base_estimator, # 修改4：使用自定义基分类器
        **params,
        random_state=42
    )

    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(save_path, 'adaboost_model.pkl'))  # 修改模型名称
    return model


# 评估函数优化 --------------------------------------------------
# def calculate_calibration_slope(y_true, y_pred_proba):
#     """数值稳定版校准斜率计算"""
#     eps = 1e-12
#     y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
#     logit_pred = np.log(y_pred_proba / (1 - y_pred_proba)).reshape(-1, 1)
#
#     try:
#         lr = LogisticRegression(fit_intercept=False, max_iter=1000)
#         lr.fit(logit_pred, y_true)
#         return lr.coef_[0][0]
#     except Exception as e:
#         print(f"校准斜率计算异常: {str(e)}")
#         return np.nan


def adaboost_test_with_metrics_ci(model, X_data, y_data, data_type, threshold=0.5, n_bootstrap=1000, ci=0.95):
    """带置信区间的模型评估"""
    print(f"\n{data_type}数据集评估结果:")
    pr=model.predict_proba(X_data)
    # 初始化存储
    bootstrap_y = []
    bootstrap_probs = []
    bootstrap_preds = []

    metrics = {
        "Accuracy": [], "Sensitivity": [], "Specificity": [],
        "F1-Score": [], "AUC": [], "PPV": [], "NPV": [],
        "Brier Score": [], "Log-Loss": [], "H-L p-value": [],
        # "Calibration Slope": []  # 新增校准斜率存储
    }

    # 主循环
    for i in range(n_bootstrap):
        # 分层重采样
        pos_idx = np.where(y_data == 1)[0]
        neg_idx = np.where(y_data == 0)[0]

        resampled_pos = resample(pos_idx, replace=True, n_samples=len(pos_idx), random_state=i)
        resampled_neg = resample(neg_idx, replace=True, n_samples=len(neg_idx), random_state=i)
        sample_idx = np.concatenate([resampled_pos, resampled_neg])

        X_sample = X_data.iloc[sample_idx] if isinstance(X_data, pd.DataFrame) else X_data[sample_idx]
        y_sample = y_data.iloc[sample_idx] if isinstance(y_data, pd.Series) else y_data[sample_idx]

        # 预测
        probas = model.predict_proba(X_sample)[:, 1]
        preds = (probas >= threshold).astype(int)
        # try:
        #     slope = calculate_calibration_slope(y_sample, probas)
        # except:
        #     slope = np.nan
        # metrics["Calibration Slope"].append(slope)

        # 计算指标
        tn, fp, fn, tp = confusion_matrix(y_sample, preds).ravel()
        metrics["Accuracy"].append(accuracy_score(y_sample, preds))
        metrics["Sensitivity"].append(recall_score(y_sample, preds))
        metrics["Specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        metrics["F1-Score"].append(f1_score(y_sample, preds))
        metrics["AUC"].append(roc_auc_score(y_sample, probas))
        metrics["PPV"].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        metrics["NPV"].append(tn / (tn + fn) if (tn + fn) > 0 else 0)
        metrics["Brier Score"].append(brier_score_loss(y_sample, probas))
        metrics["Log-Loss"].append(log_loss(y_sample, probas))
        metrics["H-L p-value"].append(hosmer_lemeshow_test(y_sample, probas))

        # 存储数据
        bootstrap_y.append(y_sample)
        bootstrap_probs.append(probas)
        bootstrap_preds.append(preds)

    # 结果分析
    results = []
    alpha = (1 - ci) / 2
    for metric, values in metrics.items():
        mean_val = np.mean(values)
        ci_low = np.percentile(values, alpha * 100)
        ci_high = np.percentile(values, (1 - alpha) * 100)
        results.append({
            "Metric": metric,
            "Mean": mean_val,
            "CI Low": ci_low,
            "CI High": ci_high
        })

    # 打印结果
    results_df = pd.DataFrame(results)
    print(f"\n{threshold}阈值下评估指标（{ci * 100}% CI）:")
    print(results_df.to_string(index=False))


    # 可视化
    plot_roc_curve_with_ci('Adaboost', bootstrap_y, bootstrap_probs, ci=ci)
    # # 修改可视化调用方式
    probas_all = model.predict_proba(X_data)[:, 1]
    plot_calibration_curve_with_ci('Adaboost', [y_data],[probas_all], ci=ci)

    plot_dca_with_ci('Adaboost', bootstrap_y, bootstrap_probs, ci=ci)

    return results_df


# 主程序 --------------------------------------------------
if __name__ == "__main__":
    # 修改5：AdaBoost参数配置
    adaboost_params = {
        'learning_rate': 0.2,
        'n_estimators': 50
    }

    # 训练模型
    adaboost_model = train_adaboost(adaboost_params, X_train, y_train)

    # 数据集配置
    datasets = [
        # ('测试集', X_test, y_test),
        ('外部验证集', X_out, y_out)
    ]

    # 评估流程
    for name, X_data, y_data in datasets:
        if X_data.shape[1] != FEATURE_COUNT:  # 特征数校验
            print(f"特征数不匹配: 预期{FEATURE_COUNT}，实际{X_data.shape[1]}")
            continue

        _ = adaboost_test_with_metrics_ci(adaboost_model, X_data, y_data, data_type=name)

        # 保存可视化结果
        plt.figure(figsize=(15, 5))
        plt.suptitle(f'{name}评估结果', y=1.02)

        plt.subplot(131)
        plot_roc_curve_with_ci('AdaBoost', [y_data], [adaboost_model.predict_proba(X_data)[:, 1]], ci=0.95)
        plt.title('ROC曲线')

        plt.subplot(132)
        plot_calibration_curve_with_ci('AdaBoost', [y_data], [adaboost_model.predict_proba(X_data)[:, 1]], ci=0.95)
        plt.title('校准曲线')

        plt.subplot(133)
        plot_dca_with_ci('AdaBoost', [y_data], [adaboost_model.predict_proba(X_data)[:, 1]], ci=0.95)
        plt.title('决策曲线')

        plt.tight_layout()
        plt.savefig(f'{name}_AdaBoost_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
