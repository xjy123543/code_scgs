# -*- coding: utf-8 -*-
"""
基于随机森林的模型评估及可视化
"""
import sys
import lightgbm as lgb
sys.path.append("D:/.spyder-py3/autosave/utils")
from dca_curve import plot_dca_with_ci
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, f1_score, roc_auc_score)
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
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
import xgboost as xgb
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 数据准备 --------------------------------------------------
# 加载数据集

credit =pd.read_csv("sorteddataset_GBM_notime.csv", encoding='gbk')
credit2 = pd.read_csv("filtered_data_GBM_notime.csv", encoding='gbk')
credit3 = pd.read_csv("排序少样本_data_lgb.csv", encoding='gbk')

# 特征和标签提取
X = credit.iloc[0:1610, :18]  # 原始特征
y = credit.iloc[0:1610, -1]
X_train = credit2.iloc[0:1371, :18]  # 预处理后的训练特征
y_train = credit2.iloc[0:1371, -1]
X_out = credit3.iloc[0:, :18]  # 外部验证集
y_out = credit3.iloc[0:, -1]

# 划分测试集
from sklearn.model_selection import train_test_split

_, X_test, _, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# 定义评估指标函数 --------------------------------------------------
def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def train_random_forest(params, X_train, y_train, save_path='models'):
    """训练随机森林模型并保存"""
    os.makedirs(save_path, exist_ok=True)
    model = lgb.LGBMClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(save_path, 'lgb_model.pkl'))
    return model
from sklearn.linear_model import LogisticRegression
# 新增校准斜率计算函数
def calculate_calibration_slope(y_true, y_pred_proba):
    # 数值稳定性处理
    y_pred_proba = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
    # 逻辑回归拟合
    logit_pred = np.log(y_pred_proba / (1 - y_pred_proba))
    lr = LogisticRegression(fit_intercept=False, penalty=None)
    lr.fit(logit_pred.reshape(-1, 1), y_true)
    return lr.coef_[0][0]

def rf_test_with_metrics_ci(model, X_data, y_data, data_type, threshold=0.5, n_bootstrap=1000, ci=0.95):
    """带置信区间的模型评估"""
    print(f"\n{data_type}数据集评估结果:")

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
    probas_all = model.predict_proba(X_data)[:, 1]
    # 可视化
    plot_roc_curve_with_ci('LightGBM', bootstrap_y, bootstrap_probs, ci=ci)
    plot_calibration_curve_with_ci('LightGBM', [y_data],[probas_all], ci=ci)

    plot_dca_with_ci('LigHtGBM', bootstrap_y, bootstrap_probs, ci=ci)

    return results_df


# 主程序 --------------------------------------------------
if __name__ == "__main__":
    # 模型参数
    rf_params = {
        'max_depth': -1,#{'learning_rate': 0.03, 'max_depth': -1, 'n_estimators': 500
        'learning_rate': 0.03,
        'n_estimators': 500,
        'scale_pos_weight': 2.18
    }
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # 训练模型
    rf_model = train_random_forest(rf_params, X_res, y_res)

    # 在三个数据集上评估
    datasets = [
        # ('训练集', X_train, y_train),
        #  ('测试集', X_test, y_test),
        ('外部验证集', X_out, y_out)
    ]

    for name, X_data, y_data in datasets:
        # 检查特征一致性
        if X_data.shape[1] != X_train.shape[1]:
            print(f"警告：{name}特征数不一致！跳过评估")
            continue

        # # 执行评估
        _ = rf_test_with_metrics_ci(rf_model, X_data, y_data, data_type=name)

        # 保存可视化结果
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plot_roc_curve_with_ci('RF', [y_data], [rf_model.predict_proba(X_data)[:, 1]], ci=0.95)
        plt.title(f'{name} ROC')

        plt.subplot(132)
        plot_calibration_curve_with_ci('RF', [y_data], [rf_model.predict_proba(X_data)[:, 1]], ci=0.95)
        plt.title(f'{name} Calibration')

        plt.subplot(133)
        plot_dca_with_ci('RF', [y_data], [rf_model.predict_proba(X_data)[:, 1]], ci=0.95)
        plt.title(f'{name} DCA')

        plt.tight_layout()
        plt.savefig(f'{name}_evaluation.png', dpi=300)
        plt.close()
