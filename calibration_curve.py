import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

import numpy as np


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




def plot_calibration_curve_with_ci(model_name, bootstrap_y_test, bootstrap_predict_pro_test, color,ci=0.95):
    """
    绘制概率校准度曲线，带置信区间，并计算 Brier 分数均值及置信区间。
    """
    bins_range = [4]
    colors = plt.cm.tab20(np.linspace(0, 1, len(bins_range)))
    offset = 20  # 向外偏移的像素值
    vline_offset = -0.05  # 竖线的下移量

    # 存储校准结果和 Brier 分数
    calibration_results = {n_bins: {"fraction_of_positives": [], "mean_predicted_value": []} for n_bins in bins_range}
    brier_scores = []

    for i in range(len(bootstrap_predict_pro_test)):
        y_true = bootstrap_y_test[i]
        positive_weight = 1 / np.mean(y_true)
        negative_weight = 1 / (1 - np.mean(y_true))
        custom_weights = np.where(y_true == 1, positive_weight, negative_weight)

        y_scores = bootstrap_predict_pro_test[i]

        # 计算 Brier 分数
        brier_scores.append(brier_score_loss(y_true, y_scores))

        for n_bins in bins_range:
            # fraction_of_positives, mean_predicted_value = calibration_curve(
            #     y_true, y_scores, n_bins=n_bins, strategy="uniform", sample_weight=custom_weights
            # )
            fraction_of_positives, mean_predicted_value = manual_calibration(y_true, y_scores, n_bins=n_bins, sample_weight=custom_weights)
            calibration_results[n_bins]["fraction_of_positives"].append(fraction_of_positives)
            calibration_results[n_bins]["mean_predicted_value"].append(mean_predicted_value)

    # 计算 Brier 分数均值和置信区间
    mean_brier = np.mean(brier_scores)
    lower_brier = np.percentile(brier_scores, (1 - ci) / 2 * 100)
    upper_brier = np.percentile(brier_scores, (1 + ci) / 2 * 100)

    # 绘制图像
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', offset))
    ax.spines['left'].set_bounds(0, 1)

    for i, n_bins in enumerate(bins_range):
        fractions = np.array(calibration_results[n_bins]["fraction_of_positives"])
        means = np.array(calibration_results[n_bins]["mean_predicted_value"])

        # 计算均值和置信区间
        mean_fraction = np.mean(fractions, axis=0)
        mean_predicted = np.mean(means, axis=0)

        # 绘制均值校准曲线
        ax.plot(
            mean_predicted, mean_fraction, marker="o", markerfacecolor="none",
            markeredgecolor=color, lw=1.5, color=color
        )

        # 绘制竖线辅助线
        for x in mean_predicted:
            ax.vlines(x, vline_offset, 0, color="blue", lw=2, transform=ax.transData)

        # 添加图例，显示 Brier 分数信息
        ax.legend(
            loc="best", fontsize=20,
            title_fontsize=15,
            title=f"{model_name} Brier: {mean_brier:.4f} [{lower_brier:.4f}, {upper_brier:.4f}]"
        )

    # 绘制参考线
    ax.plot([0, 1], [0, 1], linestyle="--", color="lightgray", lw=1)

    # 设置轴标签、范围等
    ax.set_xlabel("Predicted Probability", fontsize=20)
    ax.set_ylabel("Observed Frequency", fontsize=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(vline_offset, 1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()