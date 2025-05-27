import numpy as np
from scipy.stats import chi2


def hosmer_lemeshow_test(y_true, y_pred, num_groups=10):
    """
    Hosmer-Lemeshow 检验
    同时考虑正例和负例的分布。
    """

    # 将 y_true 和 y_pred 合并，并按预测值排序
    data = np.column_stack((y_true, y_pred))
    data = data[data[:, 1].argsort()]  # 按预测值排序

    # 提取排序后的 y_true 和 y_pred
    y_true_sorted = data[:, 0]
    y_pred_sorted = data[:, 1]

    # 分组
    percentiles = np.percentile(y_pred_sorted, np.linspace(0, 100, num_groups))
    groups = np.digitize(y_pred_sorted, bins=percentiles) - 1

    # 初始化统计量
    observed_pos = np.zeros(num_groups)
    expected_pos = np.zeros(num_groups)
    observed_neg = np.zeros(num_groups)
    expected_neg = np.zeros(num_groups)

    for i in range(num_groups):
        group_idx = (groups == i)

        # 正类和负类的索引
        pos_idx = group_idx & (y_true_sorted == 1)  # 实际正类的索引
        neg_idx = group_idx & (y_true_sorted == 0)  # 实际负类的索引

        # 分别计算实际值和预测值的正类和负类求和
        observed_pos[i] = np.sum(y_true_sorted[pos_idx])  # 实际正例
        expected_pos[i] = np.sum(y_pred_sorted[pos_idx])  # 预测正例
        observed_neg[i] = np.sum(y_true_sorted[neg_idx])  # 实际负例
        expected_neg[i] = np.sum(y_pred_sorted[neg_idx])  # 预测负例

    # 计算卡方统计量
    chi_square_statistic = 0
    for o_p, e_p, o_n, e_n in zip(observed_pos, expected_pos, observed_neg, expected_neg):
        if e_p > 0:
            chi_square_statistic += (o_p - e_p) ** 2 / e_p
        if e_n > 0:
            chi_square_statistic += (o_n - e_n) ** 2 / e_n

    # 自由度
    df = num_groups - 2

    # 计算p值
    p_value = 1 - chi2.cdf(chi_square_statistic, df)

    return p_value

