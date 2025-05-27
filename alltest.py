#这个程序运行会有些慢，因为计算量比较大
import sys
import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, recall_score, f1_score, roc_auc_score,
                             log_loss, brier_score_loss, confusion_matrix)
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from calibration_curve import plot_calibration_curve_with_ci
from dca_curve import plot_dca_with_ci
from roc_curve import plot_roc_curve_with_ci
from hl_test import hosmer_lemeshow_test

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def manual_calibration(y_true, y_scores, n_bins, sample_weight):
    """
    手动实现概率校准逻辑，模拟 calibration_curve 逻辑。
    返回:
    fraction_positives: 每个 bin 内实际阳性率 加权重
    mean_predicted: 每个 bin 内平均预测概率
    """

    # 设定分箱边界
    bin_edges = np.linspace(0, 1, n_bins + 1)  # 均匀分为 n_bins 个箱子
    mean_predicted = []
    fraction_positives = []

    # 遍历每个分箱
    for i in range(n_bins):
        # 获取当前 bin 区间内的样本索引
        in_bin = (y_scores >= bin_edges[i]) & (y_scores < bin_edges[i + 1])

        # 如果当前 bin 内样本数为空，跳过
        if np.sum(in_bin) == 0:
            continue

        # 计算当前 bin 内的权重化实际阳性率
        weighted_positives = np.sum(y_true[in_bin] * sample_weight[in_bin])
        weighted_total = np.sum(sample_weight[in_bin])

        if weighted_total > 0:
            fraction_positives.append(weighted_positives / weighted_total)  # 权重化阳性比例
            mean_predicted.append(np.mean(y_scores[in_bin]))  # 平均预测概率

    return np.array(fraction_positives), np.array(mean_predicted)


def plot_calibration_curve_with_ci(model_name, bootstrap_y_test, bootstrap_predict_pro_test, color, ci=0.95, ax=None):
    """绘制概率校准度曲线，带置信区间（保持原始样式+ax参数支持）"""
    # 创建坐标轴对象
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        show_plot = True
    else:
        show_plot = False

    # 保持原始样式参数
    bins_range = [4]
    offset = 20
    vline_offset = -0.05
    marker_style = {"marker": "o", "markerfacecolor": "none", "markeredgecolor": color}
    vline_style = {"color": "blue", "lw": 2}

    # 数据计算逻辑（保持原有逻辑）
    calibration_results = {n_bins: {"fraction_of_positives": [], "mean_predicted_value": []} for n_bins in bins_range}
    brier_scores = []

    for i in range(len(bootstrap_predict_pro_test)):
        y_true = bootstrap_y_test[i]
        positive_weight = 1 / np.mean(y_true)
        negative_weight = 1 / (1 - np.mean(y_true))
        custom_weights = np.where(y_true == 1, positive_weight, negative_weight)
        y_scores = bootstrap_predict_pro_test[i]

        # 计算Brier分数
        brier_scores.append(brier_score_loss(y_true, y_scores))

        # 校准曲线计算
        for n_bins in bins_range:
            fraction_of_positives, mean_predicted_value = manual_calibration(
                y_true, y_scores, n_bins=n_bins, sample_weight=custom_weights
            )
            calibration_results[n_bins]["fraction_of_positives"].append(fraction_of_positives)
            calibration_results[n_bins]["mean_predicted_value"].append(mean_predicted_value)

    # 结果处理
    mean_brier = np.mean(brier_scores)
    lower_brier = np.percentile(brier_scores, (1 - ci) / 2 * 100)
    upper_brier = np.percentile(brier_scores, (1 + ci) / 2 * 100)

    # 绘图样式设置（保持原始坐标轴样式）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', offset))
    ax.spines['left'].set_bounds(0, 1)

    # 主绘图逻辑（保持原始绘图元素）
    for n_bins in bins_range:
        fractions = np.array(calibration_results[n_bins]["fraction_of_positives"])
        means = np.array(calibration_results[n_bins]["mean_predicted_value"])
        mean_fraction = np.mean(fractions, axis=0)
        mean_predicted = np.mean(means, axis=0)

        # 主曲线（保持原始标记样式）
        ax.plot(
            mean_predicted, mean_fraction,
            **marker_style,  # 保持空心圆标记
            lw=1.5,
            color=color
        )

        # 竖线（保持蓝色粗线样式）
        for x in mean_predicted:
            ax.vlines(x, vline_offset, 0, **vline_style)

    # 参考线（保持浅灰色虚线）
    ax.plot([0, 1], [0, 1], linestyle="--", color="lightgray", lw=1)

    # 标签和图例（保持20号字体和标题格式）
    ax.set_xlabel("Predicted Probability", fontsize=20)
    ax.set_ylabel("Observed Frequency", fontsize=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(vline_offset, 1)

    # 保持图例样式（15号字体+标题格式）
    legend = ax.legend(
        title=f"{model_name} Brier: {mean_brier:.4f} [{lower_brier:.4f}, {upper_brier:.4f}]",
        loc="best",
        fontsize=15,
        title_fontsize=15
    )

    # 保持刻度标签样式
    ax.tick_params(axis='both', labelsize=15)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    main_line = ax.plot(mean_predicted, mean_fraction,
                        **marker_style,
                        lw=1.5,
                        color=color,
                        label=model_name)[0]
    return main_line  # 原返回ax改为返回线条对象


def plot_dca_with_ci(model_name, bootstrap_y_test, bootstrap_predict_pro_test, color, ci=0.95, ax=None):
    # 修复1：完整保留数据计算逻辑
    thresholds = np.linspace(0, 1, 101)
    standardized_net_benefit_list = []

    # 完整的数据处理流程
    for i in range(len(bootstrap_predict_pro_test)):
        y_true = np.array(bootstrap_y_test[i])
        positive_weight = 1 / (np.mean(y_true) + 1e-6)  # 添加极小值防止除零
        negative_weight = 1 / (1 - np.mean(y_true) + 1e-6)

        y_scores = np.array(bootstrap_predict_pro_test[i])
        y_pred_matrix = (y_scores[:, np.newaxis] >= thresholds).astype(int)

        tp = np.sum((y_pred_matrix == 1) & (y_true[:, np.newaxis] == 1), axis=0)
        fp = np.sum((y_pred_matrix == 1) & (y_true[:, np.newaxis] == 0), axis=0)
        tn = np.sum((y_pred_matrix == 0) & (y_true[:, np.newaxis] == 0), axis=0)
        fn = np.sum((y_pred_matrix == 0) & (y_true[:, np.newaxis] == 1), axis=0)

        net_benefit = positive_weight * (tp / (tp + fn + tn + fp + 1e-6)) - \
                      negative_weight * (fp / (tp + fn + tn + fp + 1e-6)) * (thresholds / (1 - thresholds + 1e-6))
        standardized_net_benefit_list.append(net_benefit)

    # 转换为numpy数组
    standardized_net_benefit_list = np.array(standardized_net_benefit_list)

    # 确保统计量计算
    mean_benefit = np.nanmean(standardized_net_benefit_list, axis=0)  # 使用nan安全计算
    lower_benefit = np.nanpercentile(standardized_net_benefit_list, (1 - ci) / 2 * 100, axis=0)
    upper_benefit = np.nanpercentile(standardized_net_benefit_list, (1 + ci) / 2 * 100, axis=0)

    # 计算benefit_all的逻辑
    prevalence_overall = np.mean([np.mean(y) for y in bootstrap_y_test])
    benefit_all = (
            (1 / (prevalence_overall + 1e-6)) * prevalence_overall -
            (1 / (1 - prevalence_overall + 1e-6)) * (1 - prevalence_overall) * (thresholds / (1 - thresholds + 1e-6))
    )

    # 坐标轴处理逻辑优化
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        standalone_plot = True
    else:
        standalone_plot = False
        # 当传入ax时，继承已有坐标轴设置
        ax.set_xlabel("High Risk Threshold", fontsize=20)
        ax.set_ylabel("Standardized Net Benefit", fontsize=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1)

    # 绘图元素标签优化
    main_line, = ax.plot(thresholds, mean_benefit,
                         label=model_name,  # 仅设置模型名称标签
                         color=color,
                         lw=2)

    # 置信区间不单独设置标签（避免图例重复）
    ax.fill_between(thresholds, lower_benefit, upper_benefit,
                    color=color, alpha=0.2)

    # 基准线智能绘制（仅首次绘制）
    if not hasattr(ax, '_dca_baseline_drawn'):  # 通过属性标记防止重复绘制
        ax.plot(thresholds, [0] * len(thresholds),
                linestyle="-", lw=2, color="gray",
                label='None')
        ax.plot(thresholds, benefit_all,
                linestyle="-", lw=1, color="lightgray",
                label='All')
        ax._dca_baseline_drawn = True  # 标记已绘制基线

    # 坐标轴设置解耦
    if standalone_plot:
        # 次坐标轴设置
        cost_benefit_labels = ["1:100", "1:4", "2:3", "3:2", "4:1", "100:1"]
        cost_benefit_positions = np.linspace(0, 1, len(cost_benefit_labels))
        ax_cb = ax.secondary_xaxis(-0.125)
        ax_cb.set_xticks(cost_benefit_positions)
        ax_cb.set_xticklabels(cost_benefit_labels)
        ax_cb.set_xlabel("Cost:Benefit Ratio", fontsize=20)

        # 独立绘图时立即显示
        ax.legend(loc="upper right", fontsize=12)
        plt.tight_layout()
        plt.show()

    return main_line  # 确保返回主线条对象


def plot_roc_curve_with_ci(model_name, bootstrap_y_test, bootstrap_predict_pro_test, color, ci=0.95, ax=None):
    """绘制带置信区间的ROC曲线（保持统一样式）"""
    # 创建坐标轴对象
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        show_plot = True
    else:
        show_plot = False

    # 数据计算逻辑保持不变
    thresholds = np.linspace(0, 1, 100)
    fpr_list = []
    tpr_list = []
    auc_list = []

    for i in range(len(bootstrap_predict_pro_test)):
        y_true = np.array(bootstrap_y_test[i])
        y_scores = np.array(bootstrap_predict_pro_test[i])
        y_pred_matrix = (y_scores[:, np.newaxis] >= thresholds).astype(int)

        tp = np.sum((y_pred_matrix == 1) & (y_true[:, np.newaxis] == 1), axis=0)
        fp = np.sum((y_pred_matrix == 1) & (y_true[:, np.newaxis] == 0), axis=0)
        fn = np.sum((y_pred_matrix == 0) & (y_true[:, np.newaxis] == 1), axis=0)
        tn = np.sum((y_pred_matrix == 0) & (y_true[:, np.newaxis] == 0), axis=0)

        fpr = fp / (fp + tn + 1e-10)
        tpr = tp / (tp + fn + 1e-10)
        auc_score = roc_auc_score(y_true, y_scores)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc_score)

    # 结果处理
    mean_fpr = np.mean(fpr_list, axis=0)
    mean_tpr = np.mean(tpr_list, axis=0)
    mean_auc = np.mean(auc_list)
    lower_auc = np.percentile(auc_list, (1 - ci) / 2 * 100)
    upper_auc = np.percentile(auc_list, (1 + ci) / 2 * 100)
    lower_tpr = np.percentile(tpr_list, (1 - ci) / 2 * 100, axis=0)
    upper_tpr = np.percentile(tpr_list, (1 + ci) / 2 * 100, axis=0)

    # 统一绘图样式
    ax.plot(
        mean_fpr, mean_tpr,
        color=color, lw=2,
        label=f"{model_name} AUC: {mean_auc:.4f} [{lower_auc:.4f}, {upper_auc:.4f}]"
    )
    ax.fill_between(
        mean_fpr, lower_tpr, upper_tpr,
        color=color, alpha=0.2,
        label=f"{ci * 100:.1f}% CI"  # 添加置信区间标签
    )

    # 参考线样式调整
    ax.plot([0, 1], [0, 1],
            color='lightgray',
            linestyle='-',  # 改为实线
            lw=1)

    # 统一标签样式
    ax.set_xlabel('False Positive Rate', fontsize=20)
    ax.set_ylabel('True Positive Rate', fontsize=20)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # 统一刻度字体
    ax.tick_params(axis='both', labelsize=15)

    # 图例样式
    ax.legend(
        loc="lower right",
        fontsize=15,
        frameon=True,
        framealpha=0.8
    )

    # 修改返回值：返回主线条对象
    main_line, = ax.plot(mean_fpr, mean_tpr,
                         color=color, lw=2,
                         label=f"{model_name}")
    return main_line  # 原返回ax改为返回线条对象


# --------------------------- 数据准备 ---------------------------
rf_credit = pd.read_csv("sorteddataset_Decisiontree_notime.csv", encoding='gbk')
rf_credit2 = pd.read_csv("filtered_data_Decisiontree_notime.csv", encoding='gbk')
rf_credit3 = pd.read_csv("排序少样本_data.csv", encoding='gbk')

X_rf = rf_credit.iloc[0:1610, :18]
y_rf = rf_credit.iloc[0:1610, -1]
X_train_rf = rf_credit2.iloc[0:1371, :18]
y_train_rf = rf_credit2.iloc[0:1371, -1]
X_out_rf = rf_credit3.iloc[0:, :18]
y_out_rf = rf_credit3.iloc[0:, -1]

_, X_test_rf, _, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.1, random_state=42)
# ====================== XGBoost 数据加载 ========================
xgb_credit = pd.read_csv("sorted_data_xgb_notime.csv", encoding='gbk')
xgb_credit2 = pd.read_csv("filtered_data_xgb_notime.csv", encoding='gbk')
xgb_credit3 = pd.read_csv("排序少样本_data_xgb.csv", encoding='gbk')

X_xgb = xgb_credit.iloc[0:1610, :19]  # XGBoost特征列数为19
y_xgb = xgb_credit.iloc[0:1610, -1]
X_train_xgb = xgb_credit2.iloc[0:1371, :19]
y_train_xgb = xgb_credit2.iloc[0:1371, -1]
X_out_xgb = xgb_credit3.iloc[0:, :19]
y_out_xgb = xgb_credit3.iloc[0:, -1]

_, X_test_xgb, _, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.1, random_state=42)

# ====================== LightGBM 数据加载 ========================
lgb_credit = pd.read_csv("sorteddataset_GBM_notime.csv", encoding='gbk')
lgb_credit2 = pd.read_csv("filtered_data_GBM_notime.csv", encoding='gbk')
lgb_credit3 = pd.read_csv("排序少样本_data_lgb.csv", encoding='gbk')

X_lgb = lgb_credit.iloc[0:1610, :18]  # LightGBM特征列数为18
y_lgb = lgb_credit.iloc[0:1610, -1]
X_train_lgb = lgb_credit2.iloc[0:1371, :18]
y_train_lgb = lgb_credit2.iloc[0:1371, -1]
X_out_lgb = lgb_credit3.iloc[0:, :18]
y_out_lgb = lgb_credit3.iloc[0:, -1]

_, X_test_lgb, _, y_test_lgb = train_test_split(X_lgb, y_lgb, test_size=0.1, random_state=42)

# ====================== Adaboost 数据加载 ========================
ada_credit = pd.read_csv("sorteddataset_Decisiontree_notime.csv", encoding='gbk')
ada_credit2 = pd.read_csv("filtered_data_Decisiontree_notime.csv", encoding='gbk')
ada_credit3 = pd.read_csv("排序少样本_data.csv", encoding='gbk')

X_ada = ada_credit.iloc[0:1610, :18]  # Adaboost特征列数为15
y_ada = ada_credit.iloc[0:1610, -1]
X_train_ada = ada_credit2.iloc[0:1371, :18]
y_train_ada = ada_credit2.iloc[0:1371, -1]
X_out_ada = ada_credit3.iloc[0:, :18]
y_out_ada = ada_credit3.iloc[0:, -1]

_, X_test_ada, _, y_test_ada = train_test_split(X_ada, y_ada, test_size=0.1, random_state=42)

# ====================== Decision Tree 数据加载 ========================
dt_credit = pd.read_csv("sorteddataset_Decisiontree_notime.csv", encoding='gbk')
dt_credit2 = pd.read_csv("filtered_data_Decisiontree_notime.csv", encoding='gbk')
dt_credit3 = pd.read_csv("排序少样本_data.csv", encoding='gbk')

X_dt = dt_credit.iloc[0:1610, :15]  # 决策树特征列数为15
y_dt = dt_credit.iloc[0:1610, -1]
X_train_dt = dt_credit2.iloc[0:1371, :15]
y_train_dt = dt_credit2.iloc[0:1371, -1]
X_out_dt = dt_credit3.iloc[0:, :15]
y_out_dt = dt_credit3.iloc[0:, -1]

_, X_test_dt, _, y_test_dt = train_test_split(X_dt, y_dt, test_size=0.1, random_state=42)

# ------------------------- 评估指标函数 --------------------------
def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def calculate_calibration_slope(y_true, y_pred_proba):
    y_pred_proba = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
    logit_pred = np.log(y_pred_proba / (1 - y_pred_proba))
    lr = LogisticRegression(fit_intercept=False, penalty=None)
    lr.fit(logit_pred.reshape(-1, 1), y_true)
    return lr.coef_[0][0]


# ------------------------- 模型训练函数 --------------------------
def train_random_forest(params, X_train, y_train, save_path='models'):
    os.makedirs(save_path, exist_ok=True)
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(save_path, 'random_forest_model.pkl'))
    return model


def train_adaboost(params, X_train, y_train, save_path='models'):
    os.makedirs(save_path, exist_ok=True)
    base_estimator = DecisionTreeClassifier(max_depth=15, random_state=42)
    model = AdaBoostClassifier(estimator=base_estimator, **params, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(save_path, 'adaboost_model.pkl'))
    return model


def train_decision_tree(params, X_train, y_train, save_path='models'):
    os.makedirs(save_path, exist_ok=True)
    model = DecisionTreeClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(save_path, 'dt_model.pkl'))
    return model


def train_xgboost(params, X_train, y_train, save_path='models'):
    os.makedirs(save_path, exist_ok=True)
    model = xgb.XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(save_path, 'xgb_model.pkl'))
    return model


def train_lightgbm(params, X_train, y_train, save_path='models'):
    os.makedirs(save_path, exist_ok=True)
    model = lgb.LGBMClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(save_path, 'lgb_model.pkl'))
    return model


# ---------------------- 带置信区间的评估函数 ----------------------
def model_test_with_metrics_ci(model, X_data, y_data, data_type, model_name, threshold=0.5, n_bootstrap=1000, ci=0.95):
    print(f"\n{data_type}数据集上{model_name}评估结果:")

    bootstrap_y = []
    bootstrap_probs = []
    bootstrap_preds = []

    metrics = {
        "Accuracy": [], "Sensitivity": [], "Specificity": [],
        "F1-Score": [], "AUC": [], "PPV": [], "NPV": [],
        "Brier Score": [], "Log-Loss": [], "H-L p-value": [],
    }

    for i in range(n_bootstrap):
        pos_idx = np.where(y_data == 1)[0]
        neg_idx = np.where(y_data == 0)[0]

        resampled_pos = resample(pos_idx, replace=True, n_samples=len(pos_idx), random_state=i)
        resampled_neg = resample(neg_idx, replace=True, n_samples=len(neg_idx), random_state=i)
        sample_idx = np.concatenate([resampled_pos, resampled_neg])

        X_sample = X_data.iloc[sample_idx] if isinstance(X_data, pd.DataFrame) else X_data[sample_idx]
        y_sample = y_data.iloc[sample_idx] if isinstance(y_data, pd.Series) else y_data[sample_idx]

        probas = model.predict_proba(X_sample)[:, 1]
        preds = (probas >= threshold).astype(int)

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

        bootstrap_y.append(y_sample)
        bootstrap_probs.append(probas)
        bootstrap_preds.append(preds)

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

    results_df = pd.DataFrame(results)
    print(f"\n{threshold}阈值下评估指标（{ci * 100}% CI）:")
    print(results_df.to_string(index=False))

    probas_all = model.predict_proba(X_data)[:, 1]
    return bootstrap_y, bootstrap_probs, bootstrap_preds, probas_all


# ------------------------ 模型参数配置 -------------------------
rf_params = {
    'max_depth': 15,
    'max_features': 12,
    'n_estimators': 80,
    'class_weight': {0: 1, 1: 2.19}
}

adaboost_params = {
    'learning_rate': 0.2,
    'n_estimators': 50
}

dt_params = {
    'max_depth': None,
    'max_features': 10,
    'class_weight': {0: 1, 1: 2.18}
}

xgb_params = {
    'max_depth': 31,
    'learning_rate': 0.03,
    'n_estimators': 500,
    'scale_pos_weight': 5.6
}

lgb_params = {
    'max_depth': -1,
    'learning_rate': 0.03,
    'n_estimators': 500,
    'scale_pos_weight': 2.18
}

# ------------------------- 模型训练 ---------------------------
print("开始模型训练...")
rf_model = train_random_forest(rf_params, X_train_rf, y_train_rf)
adaboost_model = train_adaboost(adaboost_params, X_train_ada, y_train_ada)
dt_model = train_decision_tree(dt_params, X_train_dt, y_train_dt)
xgb_model = train_xgboost(xgb_params, X_train_xgb, y_train_xgb)
lgb_model = train_lightgbm(lgb_params, X_train_lgb, y_train_lgb)
print("模型训练完成！")

# ---------------------- 评估与可视化配置 ------------------------
datasets = [
    ('测试集')

]# 外部验证时改成datasets = [('外部验证集')]

model_colors = {
    'Random Forest': '#1f77b4',
    'Adaboost': '#ff7f0e',
    'Decision Tree': '#2ca02c',
    'XGBoost': '#d62728',
    'LightGBM': '#9467bd'
}

from matplotlib.legend_handler import HandlerTuple, HandlerPatch
# --------------------- 合并绘图核心函数 -------------------------
def plot_combined_curves(name, results, curve_type):
    """通用曲线合并绘制函数（修正图例处理）"""
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    handles = []
    labels = []

    # 配置参数
    if curve_type == 'roc':
        plot_func = plot_roc_curve_with_ci
        data_selector = lambda r: (r[0], r[1])
        baseline = lambda: ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, lw=2)
        title = f'{name} ROC曲线对比'
        xlabel = 'False Positive Rate'
        ylabel = 'True Positive Rate'
    elif curve_type == 'calibration':
        plot_func = plot_calibration_curve_with_ci
        data_selector = lambda r: ([y_test_lgb], [r[3]])
        baseline = lambda: ax.plot([0, 1], [0, 1], 'k:', alpha=0.6, lw=2)
        title = f'{name} 校准曲线对比'
        xlabel = 'Predicted Frequency'
        ylabel = 'Observed Frequency'
    elif curve_type == 'dca':
        plot_func = plot_dca_with_ci
        data_selector = lambda r: (r[0], r[1])
        baseline = lambda: None
        title = f'{name} DCA曲线对比'
        xlabel = 'High Risk Threshold'
        ylabel = 'Standardized Net Benefits'

    # 绘制各模型曲线
    for model_name, result in results.items():
        line = plot_func(  # 直接获取线条对象
            model_name=model_name,
            bootstrap_y_test=data_selector(result)[0],
            bootstrap_predict_pro_test=data_selector(result)[1],
            color=model_colors[model_name],
            ci=0.95,
            ax=ax
        )
        handles.append(line)
        labels.append(model_name)

    # 绘制公共元素
    baseline()
    ax.set_title(title, fontsize=20, pad=15)
    ax.set_xlabel(xlabel, fontsize=16, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=16, labelpad=10)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3)

    ax.legend(
        handles=handles,
        labels=labels,
        loc='lower right' if curve_type == 'roc' else 'upper left',
        fontsize=12,
        title="Models",
        title_fontsize=14,
        framealpha=0.95,
        shadow=True,
        # 新增以下两个参数
        handler_map={tuple: HandlerTuple(ndivide=None)},  # 保持颜色块完整
        bbox_to_anchor=(1.25, 1)   # 调整DCA图例位置
    )

    plt.savefig(f'{name}_{curve_type}_合并对比.png', dpi=300, bbox_inches='tight')
    plt.close()



# ---------------------- 主评估流程 ------------------------------
for name in datasets:

    print(f"\n{'=' * 30} {name} 评估开始 {'=' * 30}")
    results = {}

    # 随机森林评估
    print("\n评估随机森林...")
    rf_res = model_test_with_metrics_ci(rf_model, X_test_rf, y_test_rf, name, 'Random Forest')
    #rf_res = model_test_with_metrics_ci(rf_model, X_out_rf, y_out_rf, name, 'Random Forest')
    #外部验证把test改成out
    results['Random Forest'] = rf_res

    # Adaboost评估
    print("\n评估Adaboost...")
    adaboost_res = model_test_with_metrics_ci(adaboost_model, X_test_ada, y_test_ada, name, 'Adaboost')
    #adaboost_res = model_test_with_metrics_ci(adaboost_model, X_out_ada, y_out_ada, name, 'Adaboost')
    results['Adaboost'] = adaboost_res

    # 决策树评估
    print("\n评估决策树...")
    dt_res = model_test_with_metrics_ci(dt_model, X_test_dt, y_test_dt, name, 'Decision Tree')
    #dt_res = model_test_with_metrics_ci(dt_model, X_out_dt, y_out_dt, name, 'Decision Tree')
    results['Decision Tree'] = dt_res

    # XGBoost评估
    print("\n评估XGBoost...")
    xgb_res = model_test_with_metrics_ci(xgb_model, X_test_xgb, y_test_xgb, name, 'XGBoost')
    #xgb_res = model_test_with_metrics_ci(xgb_model, X_out_xgb, y_out_xgb, name, 'XGBoost')
    results['XGBoost'] = xgb_res

    # LightGBM评估
    print("\n评估LightGBM...")
    lgb_res = model_test_with_metrics_ci(lgb_model, X_test_lgb, y_test_lgb, name, 'LightGBM')
    #lgb_res = model_test_with_metrics_ci(lgb_model, X_out_lgb, y_out_lgb, name, 'LightGBM')
    results['LightGBM'] = lgb_res

    # 合并可视化
    plot_combined_curves(name, results, 'roc')
    plot_combined_curves(name, results, 'calibration')
    plot_combined_curves(name, results, 'dca')

print("\n所有评估完成！结果文件已保存。")
