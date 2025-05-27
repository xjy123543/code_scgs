# -*- coding: utf-8 -*-
"""
基于RFE选择的特征和训练好的模型展示SHAP分析；
@author: wangxuechao
"""
import warnings
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap




plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


def perform_shap_analysis(model,
                          data_train,
                          data_test,
                          data_label_train,
                          data_label_test,
                          selected_features,
                          sample_index=0):
    """
    对模型进行 SHAP 分析并生成相关图形。

    Parameters:
    - model: 训练好的 LightGBM 模型
    - data_train: 训练集特征数据（DataFrame）
    - data_test: 测试集特征数据（DataFrame）
    - data_label_train: 训练集标签数据
    - data_label_test: 测试集标签数据
    - selected_features: 已选择的特征列表
    - sample_index: 要绘制 force plot 的样本索引，默认是 0
    """

    print("开始 SHAP 分析...")
    explainer = shap.TreeExplainer(model, data_train)

    shap_values = explainer(data_train)

    shap.plots.bar(shap_values)

    shap.plots.bar(shap_values[0])

    shap.plots.beeswarm(shap_values)

    # for feature_name in selected_features:
    #     shap.plots.scatter(shap_values[:, feature_name], color=shap_values)

    shap.plots.violin(shap_values, feature_names=selected_features)
    # shap.plots.violin(shap_values, features=data_train, feature_names=selected_features, plot_type="layered_violin")

    shap.plots.heatmap(shap_values, max_display=12)

    shap.plots.waterfall(shap_values[0])






if __name__ == "__main__":

    warnings.warn("This script is aimed to explain this model via SHAP.")


    # 设置目标类型和数据路径
    objective = 'prognosis'  # choices=['local', 'neck', 'transform', 'prognosis']
    data_path = f'{objective}'

    # 读取保存的特征名称
    selected_features_file = os.path.join(data_path, f'lightgbm_selected_features.txt')
    with open(selected_features_file, 'r', encoding='utf-8') as f:
        selected_features = [line.strip() for line in f.readlines()]  # 读取每一行并去除换行符

    # 加载数据
    train_file = os.path.join(data_path, 'train.xlsx')
    test_file = os.path.join(data_path, 'test.xlsx')
    data_train = pd.read_excel(train_file, engine='openpyxl')
    data_test = pd.read_excel(test_file, engine='openpyxl')

    # 分离特征和标签
    data_feature_train, data_label_train = data_train.iloc[:, :-2].copy(), data_train.iloc[:, -1].copy()
    data_feature_test, data_label_test = data_test.iloc[:, :-2].copy(), data_test.iloc[:, -1].copy()
    train_positive_ratio = (data_label_train.sum() / len(data_label_train)) * 100
    test_positive_ratio = (data_label_test.sum() / len(data_label_test)) * 100
    print(f"Train数据集中类别 1 的占比: {train_positive_ratio:.2f}%")
    print(f"Test数据集中类别 1 的占比: {test_positive_ratio:.2f}%")

    # 筛选特征
    data_feature_train_selected = data_train[selected_features]
    data_feature_test_selected = data_test[selected_features]



    # 模型加载
    lgb_model = joblib.load(os.path.join(f'../checkpoint/{objective}', 'lightgbm_model.pkl'))


    # SHAP 分析
    print("开始 SHAP 分析...")
    perform_shap_analysis(
        model=lgb_model,
        data_train=data_feature_train_selected,
        data_test=data_feature_test_selected,
        data_label_train=data_label_train,
        data_label_test=data_label_test,
        selected_features=selected_features,
        sample_index=0  # 可调整样本索引
    )