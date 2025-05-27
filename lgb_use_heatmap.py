# -*- coding: utf-8 -*-
"""
基于RFE选择的特征和训练好的模型展示热图分析；
@author: wangxuechao
"""
import warnings
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from utils.utils import create_color_maps



plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']



def lightgbm_test_with_metrics_ci(x_test, objective, data_type, threshold=0.5):
    lgb = joblib.load(os.path.join(f'../checkpoint/{objective}', 'lightgbm_model.pkl'))
    predict_prob_test = lgb.predict_proba(x_test)[:, 1]  # 获取正类概率
    return predict_prob_test


def normalize_rows(data):
    """逐行归一化，这里传入的数据是转置后的"""
    return data.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x, axis=1)






def heatmap_analysis(data_feature_train_selected, data_label_train, predict_prob_train,
                     data_feature_test_selected, data_label_test, predict_prob_test):
    # 创建副本以避免警告
    data_feature_train_selected = data_feature_train_selected.copy()
    data_feature_test_selected = data_feature_test_selected.copy()

    # 添加分类信息
    for df, label, prob, group in [
        (data_feature_train_selected, data_label_train, predict_prob_train, 0),
        (data_feature_test_selected, data_label_test, predict_prob_test, 1)
    ]:
        df.loc[:, 'Predicted label'] = prob
        df.loc[:, 'True label'] = label
        df.loc[:, 'Group'] = group

    # 合并训练集和测试集
    combined_data = pd.concat([data_feature_train_selected, data_feature_test_selected], axis=0, ignore_index=True)
    combined_data = combined_data.iloc[:, ::-1].sort_values(by=['True label', 'Group'], ascending=[True, True])
    heatmap_data = combined_data.T

    # 数据归一化
    heatmap_data_normalized = normalize_rows(heatmap_data)

    # 动态生成色图
    color_maps = create_color_maps(heatmap_data_normalized.shape[0])

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(heatmap_data_normalized.shape[1])  # 每列的位置

    for i, (feature_name, row) in enumerate(heatmap_data_normalized.iterrows()):
        color_map = color_maps[i]  # 获取对应的渐变色
        colors = color_map(row)  # 根据归一化值映射颜色
        ax.vlines(
            x=x_positions,
            ymin=i + 0.05,
            ymax=i + 0.95,
            color=colors,
            lw=0.5  # 线宽
        )
        # 在每一行右侧添加变量名
        ax.text(
            x=len(x_positions) + 0.5,  # 右侧边缘
            y=i + 0.5,  # 行中心
            s=feature_name,  # 变量名
            va='center',  # 垂直居中
            ha='left',  # 左对齐
            fontsize=9  # 字体大小
        )

    # 去掉四周坐标
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')  # 隐藏坐标轴框线

    # 美化和布局调整
    plt.tight_layout()
    plt.show()






if __name__ == "__main__":

    warnings.warn("This script is aimed to plot the heatmap analysis of this model.")


    # 设置目标类型和数据路径
    objective = 'prognosis'  # choices=['local', 'neck', 'transform', 'prognosis']
    data_path = f'../data/dataset/{objective}'

    # 读取保存的特征名称
    selected_features_file = os.path.join(data_path, f'lightgbm_selected_features.txt')
    with open(selected_features_file, 'r') as f:
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

    # 根据选择的特征，筛选出训练数据中相应的特征
    data_feature_train_selected = data_feature_train[selected_features]
    data_feature_test_selected = data_feature_test[selected_features]


    # 模型测试
    predict_prob_train = lightgbm_test_with_metrics_ci(data_feature_train_selected, objective, data_type='Train')
    predict_prob_test = lightgbm_test_with_metrics_ci(data_feature_test_selected, objective, data_type='Test')


    heatmap_analysis(data_feature_train_selected, data_label_train, predict_prob_train,
                          data_feature_test_selected,  data_label_test,  predict_prob_test )