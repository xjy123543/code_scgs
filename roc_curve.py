
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def plot_roc_curve_with_ci(model_name, bootstrap_y_test, bootstrap_predict_pro_test,color, ci=0.95):
    """
    绘制带置信区间的 ROC 曲线，手动控制点数。
    """
    # 定义固定的阈值范围
    thresholds = np.linspace(0, 1, 100)
    fpr_list = []
    tpr_list = []
    auc_list = []

    for i in range(len(bootstrap_predict_pro_test)):
        y_true = np.array(bootstrap_y_test[i])
        y_scores = np.array(bootstrap_predict_pro_test[i])
        y_pred_matrix = (y_scores[:, np.newaxis] >= thresholds).astype(int)  # Shape: (n_samples, num_thresholds)
        tp = np.sum((y_pred_matrix == 1) & (y_true[:, np.newaxis] == 1), axis=0)
        fp = np.sum((y_pred_matrix == 1) & (y_true[:, np.newaxis] == 0), axis=0)
        fn = np.sum((y_pred_matrix == 0) & (y_true[:, np.newaxis] == 1), axis=0)
        tn = np.sum((y_pred_matrix == 0) & (y_true[:, np.newaxis] == 0), axis=0)
        fpr = fp / (fp + tn + 1e-10)  # 避免分母为零
        tpr = tp / (tp + fn + 1e-10)
        auc_score = roc_auc_score(y_true, y_scores)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc_score)

    mean_fpr = np.mean(fpr_list, axis=0)
    mean_tpr = np.mean(tpr_list, axis=0)
    mean_auc = np.mean(auc_list)
    lower_auc = np.percentile(auc_list, (1 - ci) / 2 * 100)
    upper_auc = np.percentile(auc_list, (1 + ci) / 2 * 100)

    # 计算 TPR 的置信区间
    lower_tpr = np.percentile(tpr_list, (1 - ci) / 2 * 100, axis=0)
    upper_tpr = np.percentile(tpr_list, (1 + ci) / 2 * 100, axis=0)

    # 绘制均值 ROC 曲线
    plt.figure(figsize=(8, 8))
    plt.plot(mean_fpr, mean_tpr, color=color, lw=2,
             label=f"{model_name} AUC: {mean_auc:.4f} [{lower_auc:.4f}, {upper_auc:.4f}]")
    plt.fill_between(mean_fpr, lower_tpr, upper_tpr, color=color, alpha=0.2, label=f"{ci*100:.1f}% CI")
    plt.plot([0, 1], [0, 1], color='lightgray', linestyle='-', lw=1)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    # 调整坐标轴刻度字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.legend(loc="lower right", fontsize=15)
    plt.show()
