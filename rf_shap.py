import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import numpy as np
from scipy.stats import spearmanr, pearsonr

print(shap.__version__)

import warnings


data = pd.read_csv("sorteddataset_Decisiontree_notime.csv", encoding='gbk')
data2 = pd.read_csv("filtered_data_Decisiontree_notime.csv", encoding='gbk')
data3 = pd.read_csv("排序少样本_data.csv", encoding='gbk')

    # 特征和标签提取
X = data.iloc[0:1610, :18]  # 原始特征
y = data.iloc[0:1610, -1]
data_feature_train = data2.iloc[0:1371, :18]  # 预处理后的训练特征
data_label_train= data2.iloc[0:1371, -1]
data_feature_test = data3.iloc[0:, :18]  # 外部验证集
data_label_test= data3.iloc[0:, -1]

def rf_analyze_misclassified_samples(model, explainer,shap_values, data_test_feature, data_label_test, threshold=0.5,data_feature_train_selected=data_feature_train):
    """
    完全重构的随机森林SHAP分析函数
    """
    # ====== 动态计算SHAP值 ======
    # 使用训练数据作为背景，计算测试样本的SHAP值


    # ====== 统一SHAP值格式 ======
    if isinstance(shap_values, list):  # 多分类情况
        class_index = 1  # 分析第二类
        shap_array = np.array(shap_values[class_index])  # 转换为numpy数组
        expected_value = explainer.expected_value[class_index]
    else:  # 二分类/回归情况
        shap_array = np.array(shap_values)
        if len(shap_array.shape) == 3:  # 处理二分类多维输出
            shap_array = shap_array[:, :, 1]  # 选择正类维度
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                                   np.ndarray) else explainer.expected_value

    # ====== 预测结果计算 ======
    predictions = model.predict_proba(data_test_feature)[:, 1]
    predictions_binary = (predictions > threshold).astype(int)
    misclassified = predictions_binary != data_label_test.values

    # ====== 样本分析 ======
    mis_samples = []
    correct_samples = []

    for idx in range(data_test_feature.shape[0]):
        sample_data = {
            'index': idx,
            'true_label': data_label_test.iloc[idx],
            'predicted_label': predictions_binary[idx],
            'shap_values': shap_array[idx],
            'features': data_test_feature.iloc[idx]
        }
        (mis_samples if misclassified[idx] else correct_samples).append(sample_data)

    # ====== 交互效应分析 ======
    try:
        age_idx = data_test_feature.columns.get_loc('年龄')
        high_idx = data_test_feature.columns.get_loc('病理类型-高分化粘表')

        # 验证SHAP维度
        if len(shap_array.shape) == 2:
            shap.dependence_plot(age_idx, shap_array[:, age_idx], data_test_feature,
                                 interaction_index=high_idx, show=False)
            plt.title("年龄与病理类型交互效应")
    except Exception as e:
        print(f"绘图失败: {str(e)}")

    # ====== 特征分析 ======
    # print("\n错误分类样本分析:")
    # for sample in mis_samples[:5]:  # 显示前5个错误样本
    #     print(f"\n样本 {sample['index']}:")
    #     print(f"真实标签: {sample['true_label']}, 预测: {sample['predicted_label']}")
    #     print(f"年龄: {sample['features']['age']}岁")
    #     print(f"病理类型: {'高分化' if sample['features']['type_4'] else '其他'}")

    # ====== 特征重要性对比 ======
    if mis_samples:
        mis_importance = np.abs([s['shap_values'] for s in mis_samples]).mean(axis=0)
        df_importance = pd.DataFrame({
            '特征': data_test_feature.columns,
            '错误样本影响': mis_importance,
        })
        print("\n错误样本特征重要性排序:")
        print(df_importance.sort_values('错误样本影响', ascending=False).head(10))

def rf_perform_shap_analysis_with_decision_line(model, explainer, data_test_feature, data_label_test):
    """
    最终修正版：统一SHAP值处理逻辑
    """
    # ====== 动态计算SHAP值 ======
    try:
        shap_values = explainer.shap_values(data_test_feature)
    except Exception as e:
        print(f"SHAP计算失败: {str(e)}")
        return

    # ====== 统一SHAP值格式 ======
    if isinstance(shap_values, list):
        # 多分类场景：选择目标类别（例如索引1）
        class_index = 1
        shap_values_plot = np.array(shap_values[class_index])
        expected_value = explainer.expected_value[class_index]
    else:
        # 二分类场景处理
        if len(shap_values.shape) == 3:
            # 处理sklearn的二分类三维输出 (n_samples, n_features, 2)
            shap_values_plot = shap_values[:, :, 1]  # 选择正类
            expected_value = explainer.expected_value[1]
        else:
            # 直接使用二维数组
            shap_values_plot = shap_values
            expected_value = explainer.expected_value

    # ====== 预测逻辑 ======
    predictions = model.predict_proba(data_test_feature)[:, 1]
    predictions_binary = (predictions > 0.5).astype(int)
    misclassified = predictions_binary != data_label_test.values

    # ====== 决策图绘制 ======
    shap.decision_plot(
        base_value=expected_value,
        shap_values=shap_values_plot,
        features=data_test_feature.values,
        feature_names=data_test_feature.columns.tolist(),
        highlight=misclassified
    )

def rf_verify_shap_correctness(model, data_train_feature, data_test_feature, sample_index):
    """
    使用shap.Explainer验证随机森林的SHAP正确性
    保持原有变量名和逻辑结构
    """
    # 保持原有数据获取方式
    sample = data_test_feature.iloc[[sample_index]]
    f_x = model.predict_proba(sample)[:, 1][0]  # 保持二分类假设

    # 修改为通用Explainer，添加随机森林专用参数
    explainer = shap.Explainer(
        model,
        masker=data_train_feature,  # 替换原来的data参数
        feature_perturbation="interventional",  # 关键参数
        model_output="probability"
    )

    # 保持原有SHAP值计算逻辑
    shap_values = explainer(sample)

    # ====== 新增二分类expected_value处理 ======
    if isinstance(shap_values, list):  # 多分类
        class_index = 1
        shap_values_sample = shap_values[class_index].values[0]
        y_base = explainer.expected_value[class_index]
    else:
        # 处理二分类数组结构
        if isinstance(explainer.expected_value, np.ndarray) and len(explainer.expected_value) == 2:
            y_base = explainer.expected_value[1]  # 明确选择正类基准值
        else:
            y_base = explainer.expected_value  # 回归或单分类情况

        # 确保SHAP值维度正确
        if len(shap_values.values.shape) == 3:  # 处理二分类多维输出
            shap_values_sample = shap_values.values[0, :, 1]  # 选择第二类SHAP值
        else:
            shap_values_sample = shap_values.values[0]

    # ====== 强制转换为标量 ======
    shap_sum = float(y_base) + float(shap_values_sample.sum())

    # ====== 保持原有输出 ======
    print(f"SHAP 计算值 (E[f(x)] + ∑SHAP): {shap_sum:.6f}")  # 现在可以正确格式化
    # 保持原有输出格式
    print(f"模型的预测值 f(x): {f_x:.6f}")
    print(f"SHAP值类型: {type(shap_values)}")
    print(f"SHAP值形状: {np.shape(shap_values)}")
    print(f"样本形状: {sample.shape}")
    print(f"SHAP 计算值 (E[f(x)] + ∑SHAP): {shap_sum:.6f}")
    print(f"实际差值: {abs(f_x - shap_sum):.6f}")  # 新增差值显示



def rf_perform_shap_analysis(model, data_train_feature, data_test_feature, sample_index=0):
    # """
    # 对模型进行 SHAP 分析并生成相关图形。
    # """
    #
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values( data_test_feature,)
    # # # 选择第二个类别（shap_values[1]），即类别2的 SHAP 值
    shap_values_class2 = shap_values[:, :, 1]  # 选择类别2的 SHAP 值
    shap.summary_plot(shap_values_class2,  data_test_feature,)
    shap.summary_plot(shap_values_class2,  data_test_feature,plot_type='bar')


    # ====== 2. 多分类兼容处理 ======
    if len(shap_values.shape) == 3:  # 检查多分类情况 (samples, features, classes)
        # 提取目标类别的完整解释对象
        class_shap = shap_values[..., 1].copy()  # 保持Explanation类型
        base_value = explainer.expected_value[1]
    else:
        class_shap = shap_values
        base_value = explainer.expected_value

    # ====== 3. 热力图数据包装 ======
    heatmap_data = shap.Explanation(
        values=class_shap.values if hasattr(class_shap, 'values') else class_shap,
        base_values=np.full((data_test_feature.shape[0],), float(base_value)),
        data=data_test_feature.values,
        feature_names=data_test_feature.columns.tolist()
    )
    plt.figure(figsize=(12, 8))
    shap.plots.heatmap(
        heatmap_data,  # 必须传入Explanation对象
        max_display=12,
        instance_order=heatmap_data.abs.sum(1),
        feature_values=heatmap_data.abs.mean(0),

    )
    """
        SHAP 0.46.0 多分类瀑布图专用修复
        """

    # ====== 3. 瀑布图正确调用方式 ======
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(
        heatmap_data[sample_index],  # 必须传入Explanation对象切片
        max_display=10,

    )
    plt.title(f"样本 {sample_index} 特征贡献瀑布图")
    plt.show()
    """
        通用解释器多分类兼容方案 (SHAP ≥ 0.40)
        """
    # ====== 1. 生成Explanation对象 ======
    explainer = shap.Explainer(model)
    shap_exp = explainer(data_test_feature)  # 关键：直接调用解释器

    # ====== 2. 多分类数据结构处理 ======
    if len(shap_exp.shape) == 3:  # 多分类检测 (samples, features, classes)
        # 提取目标类别的SHAP值（降为二维）
        class_values = shap_exp.values[..., 1]  # (n_samples, n_features)
        # 获取对应类别的基值（每个样本的基值相同）
        class_base = shap_exp.base_values[:, 1]  # (n_samples,)
    else:
        class_values = shap_exp.values
        class_base = shap_exp.base_values

    # ====== 3. 决策图维度修正 ======
    # 单个样本的基值必须为标量
    sample_base = class_base[sample_index].item()  # 从数组转标量
    #
    # ====== 4. 可视化执行 ======
    plt.figure(figsize=(10, 6))
    shap.decision_plot(
        base_value=sample_base,
        shap_values=class_values[sample_index],
        features=data_test_feature.iloc[sample_index:sample_index + 1],  # 保持DataFrame结构
        feature_names=data_test_feature.columns.tolist(),

    )
    shap.decision_plot(
        base_value=sample_base,
        shap_values=class_values[:],
        features=data_test_feature.iloc[:],  # 保持DataFrame结构
        feature_names=data_test_feature.columns.tolist(),

    )

    plt.show()
    # 5. SHAP特征依赖图：展示一个特征的变化如何影响模型输出
    feature_index = 0  # 选择要展示的特征的索引，可以根据需要修改
    shap.dependence_plot(
        ind='年龄',  # 目标特征名（需与DataFrame列名一致）
        shap_values=shap_exp.values[..., 1],  # 指定类别的二维SHAP值
        features=data_test_feature,  # 原始特征数据DataFrame
        interaction_index=None,  # 禁用自动交互检测    # 绘制年龄特征的交互图（自动检测最强交互）

        feature_names=data_test_feature.columns.tolist()  # 显式传递特征名
    )
    #重建单类别Explanation对象（保留所有特征）
    class2_exp = shap.Explanation(
        values=class_values,
        base_values=class_base,
        feature_names=data_test_feature.columns.tolist(),  # 确保特征名正确
        data=data_test_feature.values
    )

    shap.plots.scatter(
        class2_exp[:, "年龄"],  # 指定主特征
        color=class2_exp,  # 自动选择交互特征

    )
    plt.title("年龄特征SHAP值与交互效应")
    plt.tight_layout()
    plt.show()
    #
    # ====== 7. 蜂群图修正 ======
    # 重建基值为标量（蜂群图特殊要求）
    beeswarm_exp = shap.Explanation(
        values=class_values,
        base_values=class_base.mean(),  # 标量基值
        feature_names=data_test_feature.columns.tolist(),
        data=data_test_feature.values
    )

    shap.plots.beeswarm(beeswarm_exp)
    plt.title("类别2特征全局影响分布")
    plt.tight_layout()
    plt.show()
def rf_evaluate_faithfulness_correlation(model, data_train_feature, data_test_feature, n=10):
    """
    最终修复版本：适配SHAP最新API，解决参数传递问题
    关键修改点：
    1. 使用正确的TreeExplainer初始化方式
    2. 规范SHAP值处理逻辑
    """
    # ====== 1. 初始化SHAP解释器 ======
    explainer = shap.TreeExplainer(  # 直接使用专用解释器
        model=model,
        data=data_train_feature,  # 正确参数名
        feature_names=data_train_feature.columns.tolist(),
        model_output='probability'  # 正确的输出类型参数
    )

    # ====== 2. 计算SHAP值 ======
    shap_values = explainer.shap_values(data_test_feature)

    # 统一SHAP值格式（处理多分类）
    if isinstance(shap_values, list):
        # 选择目标类别（例如二分类选索引1）
        shap_matrix = np.array(shap_values[1])  # 假设分析正类
    else:
        # 处理sklearn的二分类三维输出
        shap_matrix = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

    # 计算平均绝对重要性
    shap_importance = np.abs(shap_matrix).mean(axis=0)

    # ====== 3. 优化扰动实验 ======
    base_pred = model.predict_proba(data_test_feature)
    spearman_scores = []

    for _ in range(n):
        delta_f = []
        # 批量生成扰动数据
        perturbed_datas = {
            col: data_test_feature[col].sample(frac=1).values
            for col in data_test_feature.columns
        }

        # 并行计算每个特征的扰动影响
        for col in data_test_feature.columns:
            perturbed_data = data_test_feature.copy()
            perturbed_data[col] = perturbed_datas[col]

            perturb_pred = model.predict_proba(perturbed_data)

            # 兼容多分类计算
            if base_pred.ndim == 2:  # 二分类
                delta = np.abs(base_pred[:, 1] - perturb_pred[:, 1]).mean()
            else:  # 多分类
                delta = np.abs(base_pred - perturb_pred).mean()

            delta_f.append(delta)

        # 计算秩相关系数
        corr, _ = spearmanr(shap_importance, delta_f)
        spearman_scores.append(corr if not np.isnan(corr) else 0)  # 处理全零情况

    # ====== 4. 结果统计 ======
    spearman_mean = np.mean(spearman_scores)
    spearman_std = np.std(spearman_scores)

    print(f"[Faithfulness] Spearman: μ = {spearman_mean:.3f} ± {spearman_std:.3f}")
    return spearman_scores



def rf_evaluate_monotonicity_ratio(model, data_train_feature, data_test_feature, perturbation_step=0.05):
    """
    随机森林的SHAP单调性评估（支持多分类）
    改进点：批量预测优化、多分类支持、数值稳定性增强
    """
    # ====== 1. 初始化SHAP解释器 ======
    explainer = shap.Explainer(
        model,
        masker=data_train_feature,
        feature_names=data_train_feature.columns.tolist(),
        model_output="probability",
        feature_perturbation="interventional"
    )

    # ====== 2. 计算SHAP值 ======
    shap_values = explainer(data_test_feature)

    # ====== 3. 准备原始预测 ======
    base_proba = model.predict_proba(data_test_feature)
    if base_proba.ndim == 2:  # 二分类
        class_index = 1
        base_proba = base_proba[:, class_index]
    else:  # 多分类
        class_index = 0  # 默认评估第一个类别
        base_proba = base_proba[..., class_index]

    # ====== 4. 批量扰动评估 ======
    valid_pairs = 0
    compliant_pairs = 0

    shap_importance = shap_values.values
    n_features = data_test_feature.shape[1]
    monotonicity_counts = 0
    total_counts = 0
    for i in range(len(data_test_feature)):
        x_original = data_test_feature.iloc[i].copy()
        f_x_original = model.predict_proba([x_original])[:, 1][0]

        for j in range(n_features):
            x_perturbed_pos = x_original.copy()
            x_perturbed_neg = x_original.copy()
            delta_x = x_original[j] * perturbation_step
            if delta_x != 0:
                x_perturbed_pos[j] += delta_x  # 正向扰动
                x_perturbed_neg[j] -= delta_x  # 负向扰动
                # 计算扰动后的预测值
                f_x_perturbed_pos = model.predict_proba([x_perturbed_pos])[:, 1][0]
                f_x_perturbed_neg = model.predict_proba([x_perturbed_neg])[:, 1][0]
                # 获取原始 SHAP 方向
                shap_value_original = shap_importance[i, j,1]
                # 判断 SHAP 解释是否符合单调性假设
                if shap_value_original > 0:
                    total_counts += 1
                    if f_x_perturbed_pos >= f_x_original >= f_x_perturbed_neg:
                        monotonicity_counts += 1
                elif shap_value_original < 0:
                    total_counts += 1
                    if f_x_perturbed_pos <= f_x_original <= f_x_perturbed_neg:
                        monotonicity_counts += 1
    monotonicity_ratio = monotonicity_counts / total_counts
    print(f"Monotonicity Ratio: {monotonicity_ratio:.4f}")


def rf_compute_explanation_complexity(model, data_train_feature, data_test_feature):
    """
    计算整个数据集的 SHAP 解释复杂度（基于熵）。熵越低，说明解释越好理解。
    因为此时SHAP值主要高度依赖少量特征，说明预测结果仅由少量特征控制，便于理解。
    """
    # 计算 SHAP 值
    explainer = shap.Explainer(model, data_train_feature, model_output="probability")
    shap_values = explainer(data_test_feature).values  # 获取所有样本的 SHAP 值

    complexities = []
    for shap_values_sample in shap_values:
        # 计算 SHAP 值的归一化贡献度
        abs_shap_values = np.abs(shap_values_sample)
        shap_sum = np.sum(abs_shap_values)
        if shap_sum == 0:
            complexities.append(0)  # 避免 log(0) 计算错误
            continue
        P_g = abs_shap_values / shap_sum  # 归一化成概率分布

        # 计算熵（Entropy）
        complexity = -np.sum(P_g * np.log2(P_g + 1e-9))  # 加入 1e-9 避免 log(0)
        complexities.append(complexity)

    print(f"Complexity: Mean={np.mean(complexities):.4f}, Std={np.std(complexities):.4f}")

def rf_compute_shap_stability(model, data_train_feature, data_test_feature, r=0.1, num_samples=10):
    """
    计算数据集中所有样本 SHAP 解释的最大敏感度（Max Sensitivity）的均值和标准差
    使用增广扰动分析（ADA）形式：
    - 对每个扰动样本根据偏离原始样本的程度设置权重
    :return: 最大敏感度的均值和标准差
    """
    explainer = shap.Explainer(model, data_train_feature, model_output="probability")

    shap_values = explainer(data_test_feature)

    sensitivities = []

    # 获取特征的总体方差和均值
    feature_variances = data_test_feature.var()
    feature_means = data_test_feature.mean()

    for sample_index in range(len(data_test_feature)):
        sample = data_test_feature.iloc[[sample_index]].copy()
        shap_original = shap_values[sample_index].values  # 原始样本 SHAP 贡献

        # 生成邻域样本（增广扰动分析）
        neighborhood = []
        distances = []  # 存储每个扰动样本与原样本的距离

        for _ in range(num_samples):
            perturbed_sample = sample.copy()

            # 对每个特征进行扰动
            for col in data_test_feature.columns:
                if col == '年龄':  # 年龄特征在标准差范围内扰动
                    perturbed_sample[col] += np.random.normal(scale=0.1)  # 高斯扰动，标准差为0.1
                else:
                    # 基于该特征的均值设置扰动概率
                    prob = feature_means[col]  # 该特征值为1的概率
                    perturbed_sample[col] = np.random.choice([0, 1])  # 按照概率进行扰动

            # 确保扰动后的值在合理范围内
            perturbed_sample = np.clip(perturbed_sample, data_test_feature.min().values, data_test_feature.max().values)
            neighborhood.append(perturbed_sample.values)

            # 计算扰动样本与原样本的欧几里得距离（偏离程度）
            distance = np.linalg.norm(perturbed_sample.values - sample.values)
            distances.append(distance)

        # 计算邻域样本的 SHAP 值
        neighborhood_df = pd.DataFrame(np.vstack(neighborhood), columns=data_test_feature.columns)
        shap_neighborhood = explainer(neighborhood_df).values

        # 根据扰动样本与原样本的距离计算权重（偏离大的样本权重小）
        weights = np.exp(-np.array(distances))  # 使用指数衰减函数作为权重
        weights /= np.sum(weights)  # 归一化权重

        # 计算 SHAP 解释的加权偏差（L2 距离）
        weighted_shap_diff = np.sum(np.fromiter((weights[i] * np.linalg.norm(shap_original - shap_z, ord=2)
                                                 for i, shap_z in enumerate(shap_neighborhood)), dtype=float))

        sensitivities.append(weighted_shap_diff)

    # 输出敏感度均值和标准差
    print(f"Stability: Weighted Mean={np.mean(sensitivities):.4f}, Std={np.std(sensitivities):.4f}")

import pickle
def main():
    """主函数：整合所有分析流程"""
    # ================== 数据加载 ==================
    # 初始化中文显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    # 加载数据集
    try:
        data = pd.read_csv("sorteddataset_Decisiontree_notime.csv", encoding='gbk')
        data2 = pd.read_csv("filtered_data_Decisiontree_notime.csv", encoding='gbk')
        data3 = pd.read_csv("排序少样本_data.csv", encoding='gbk')
    except FileNotFoundError as e:
        print(f"文件加载失败: {str(e)}")
        return

    # 特征与标签提取
    X = data.iloc[0:1610, :18]
    y = data.iloc[0:1610, -1]
    data_feature_train = data2.iloc[0:1371, :18]
    data_label_train = data2.iloc[0:1371, -1]
    data_feature_test = data3.iloc[0:, :18]
    data_label_test = data3.iloc[0:, -1]

    # ================== 模型加载 ==================
    model_path = os.path.join('models', 'random_forest_model.pkl')
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    # ================== SHAP初始化 ==================

    explainer = shap.Explainer(model, data_feature_train, model_output="probability")
    # 构建可序列化数据
    save_path=os.path.join('models', 'shap_explainer_rf.pkl')
    save_data = {
        'explainer_class': 'Explainer',
        'explainer_data': explainer.__dict__,
        'shap_version': shap.__version__,
        'model_type': 'RandomForest',
        'model_params': model.get_params()  # 保存模型参数用于验证
    }

    # 使用最新pickle协议
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"解释器已安全保存至 {save_path} (大小: {os.path.getsize(save_path) / 1024:.2f}KB)")


    shap_values = explainer(data_feature_test).values  # 获取所有样本的 SHAP 值
    # # ================== 执行分析 ==================
    # print("\n" + "=" * 30 + " SHAP正确性验证 " + "=" * 30)
    # rf_verify_shap_correctness(model, data_feature_train, data_feature_test, sample_index=6)
    #
    # print("\n" + "=" * 30 + " 错误分类样本分析 " + "=" * 30)
    # rf_analyze_misclassified_samples(model, explainer, shap_values, data_feature_test, data_label_test)
    #
    # print("\n" + "=" * 30 + " 决策路径可视化 " + "=" * 30)
    # rf_perform_shap_analysis_with_decision_line(model, explainer, data_feature_test, data_label_test)
    #
    # print("\n" + "=" * 30 + " 保真度评估 " + "=" * 30)
    # rf_evaluate_faithfulness_correlation(model, data_feature_train, data_feature_test)
    #
    # # print("\n" + "=" * 30 + " 单调性评估 " + "=" * 30)
    # # rf_evaluate_monotonicity_ratio(model, data_feature_train, data_feature_test)
    #
    # print("\n" + "=" * 30 + " 解释复杂度 " + "=" * 30)
    # rf_compute_explanation_complexity(model, data_feature_train, data_feature_test)
    #
    # print("\n" + "=" * 30 + " 稳定性评估 " + "=" * 30)
    # rf_compute_shap_stability(model, data_feature_train, data_feature_test)

    # print("\n" + "=" * 30 + " 完整SHAP分析 " + "=" * 30)
    # rf_perform_shap_analysis(model, data_feature_train, data_feature_test)

if __name__ == "__main__":
    main()
