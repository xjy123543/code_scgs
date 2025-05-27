
import numpy as np
import matplotlib.pyplot as plt


def plot_dca_with_ci(model_name, bootstrap_y_test, bootstrap_predict_pro_test, color,ci=0.95):
    """
    绘制 DCA 决策曲线分析，并动态计算每次重采样的 prevalence，展示置信区间。
    """
    thresholds = np.linspace(0, 1, 101)  # 设置 101 个阈值
    standardized_net_benefit_list = []

    for i in range(len(bootstrap_predict_pro_test)):
        y_true = np.array(bootstrap_y_test[i])
        positive_weight = 1 / np.mean(y_true)
        negative_weight = 1 / (1 - np.mean(y_true))
        y_scores = np.array(bootstrap_predict_pro_test[i])
        y_pred_matrix = (y_scores[:, np.newaxis] >= thresholds).astype(int)  # Shape: (n_samples, num_thresholds)

        tp = np.sum((y_pred_matrix == 1) & (y_true[:, np.newaxis] == 1), axis=0)
        fp = np.sum((y_pred_matrix == 1) & (y_true[:, np.newaxis] == 0), axis=0)
        fn = np.sum((y_pred_matrix == 0) & (y_true[:, np.newaxis] == 1), axis=0)
        tn = np.sum((y_pred_matrix == 0) & (y_true[:, np.newaxis] == 0), axis=0)

        # 计算 net benefit 和 standardized net benefit
        net_benefit = positive_weight * (tp / (tp + fn + tn + fp)) - negative_weight * (fp / (tp + fn + tn + fp)) * (thresholds / (1 - thresholds + 1e-6))
        standardized_net_benefit = net_benefit
        standardized_net_benefit_list.append(standardized_net_benefit)

    standardized_net_benefit_list = np.array(standardized_net_benefit_list)  # 转换为数组
    mean_benefit = np.mean(standardized_net_benefit_list, axis=0)  # 计算均值

    # 计算置信区间
    lower_benefit = np.percentile(standardized_net_benefit_list, (1 - ci) / 2 * 100, axis=0)
    upper_benefit = np.percentile(standardized_net_benefit_list, (1 + ci) / 2 * 100, axis=0)

    # 计算 "All" 策略
    prevalence_overall = np.mean([np.mean(y) for y in bootstrap_y_test])  # 总体阳性样本比例
    benefit_all = (
        positive_weight * prevalence_overall - negative_weight * (1 - prevalence_overall) * (thresholds / (1 - thresholds + 1e-6))
    )

    # 绘制图像
    fig, ax = plt.subplots(figsize=(8, 8))
    offset = 20

    # 设置主图布局
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', offset))
    ax.spines['left'].set_bounds(0, 1)

    # 绘制 DCA 曲线及置信区间
    ax.plot(thresholds, mean_benefit, label=f"{model_name}", color=color, lw=2)
    ax.fill_between(thresholds, lower_benefit, upper_benefit, color=color, alpha=0.2, label=f"{ci*100:.1f}% CI")
    ax.plot(thresholds, [0] * len(thresholds), linestyle="-", label="None", lw=2, color="gray")
    ax.plot(thresholds, benefit_all, linestyle="-", label="All", lw=1, color="lightgray")

    # 设置轴标签、范围等
    ax.set_xlabel("High Risk Threshold", fontsize=20)
    ax.set_ylabel("Standardized Net Benefit", fontsize=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # 添加次 X 轴
    cost_benefit_labels = ["1:100", "1:4", "2:3", "3:2", "4:1", "100:1"]
    cost_benefit_positions = np.linspace(0, 1, len(cost_benefit_labels))  # 对应阈值位置
    ax_cb = ax.secondary_xaxis(-0.125)
    ax_cb.set_xticks(cost_benefit_positions)
    ax_cb.set_xticklabels(cost_benefit_labels)
    ax_cb.set_xlabel("Cost:Benefit Ratio", fontsize=20)

    # 添加图例
    ax.legend(loc="best", fontsize=15, ncol=1)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()  # 自动调整布局
    plt.show()
