# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:25:28 2024

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:17:10 2024

@author: Administrator
"""

### 导入模块
import warnings
warnings.filterwarnings('ignore')
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

#预测
def performance_reg(model, X, y, metrics_name=None):
    y_pred = model.predict(X)
    if metrics_name:
        print(metrics_name, ":")
    print("Mean Squared Error (MSE): ", mean_squared_error(y, y_pred))
    print("Root Mean Squared Error (RMSE): ", np.sqrt(mean_squared_error(y, y_pred)))
    print("Mean Absolute Error (MAE): ", mean_absolute_error(y, y_pred))
    print("R2 Score: ", r2_score(y, y_pred))
    print("----------------------------")

# 数据
# 数据
"""
Created on Wed Nov 13 21:45:36 2024

@author: Administrator
"""

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


credit = pd.read_csv("datasetnotime2.csv",encoding='gbk')


from sklearn.model_selection import cross_val_predict, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler  # 注意这里使用了imblearn库
from imblearn.under_sampling import RandomUnderSampler  # 注意这里使用了imblearn库
from imblearn.over_sampling import SMOTE

# 划分训练集和测试集
X = credit.iloc[0:1610, :-5]  # 特征
y = credit.iloc[0:1610, -1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2021, shuffle=True)


import shap
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 创建数据集
from sklearn.tree import DecisionTreeRegressor
# 训练 XGBoost 模型
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)


# 创建解释器
explainer = shap.Explainer(model, X_train, feature_perturbation="interventional" )

# 计算 SHAP 值
shap_values = explainer(X_train)
shap_interaction = explainer.shap_interaction_values(X_train)
# Display summary plot

shap.summary_plot(shap_interaction, X_train,max_display=8,show_values_in_legend=True,use_log_scale=True)


# 绘制单个样本的决策图
sample_index = 0
shap.decision_plot(explainer.expected_value, shap_values.values[sample_index], X_train.iloc[sample_index])

# 绘制多个样本的决策图
shap.decision_plot(explainer.expected_value, shap_values.values[:1448], X_train.iloc[:1448])
shap.summary_plot(shap_values, X_train)

# 确保索引重置
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

# 重新初始化解释器并计算新的 SHAP 值
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)
# 计算每个特征的最大 SHAP 值
max_shap_values = np.amax(np.abs(shap_values.values), axis=0)
print(max_shap_values)
# 重新计算平均绝对SHAP值并识别离群点
mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
mean_shap = np.mean(np.abs(shap_values.values), axis=0)
std_shap = np.std(np.abs(shap_values.values), axis=0)

# 设置阈值为平均值加上两倍标准差
thresholds = mean_shap + 7* std_shap

# 寻找超过阈值的样本索引
outliers = np.where((np.abs(shap_values.values) > thresholds).any(axis=1))[0]

# 删除离群点
X_filtered = X_train.drop(index=outliers)
X_filtered = X_filtered.drop_duplicates()
y_filtered = y_train.drop(index=outliers)
y_filtered = y_filtered.loc[X_filtered.index]  # 确保y_filtered与X_filtered索引一致

X_filtered.reset_index(drop=True, inplace=True)
y_filtered.reset_index(drop=True, inplace=True)

# 重新初始化解释器并计算过滤后的 SHAP 值
explainer_filtered = shap.Explainer(model, X_filtered)
shap_values_filtered = explainer_filtered(X_filtered)

# 绘制过滤后的数据的决策图
shap.decision_plot(explainer_filtered.expected_value, shap_values_filtered.values, X_filtered)

shap.summary_plot(shap_values_filtered.values, X_filtered)
model2 =XGBClassifier(random_state=42)
model2.fit(X_filtered, y_filtered)
# 计算 SHAP 值
shap_values2 = explainer(X_test)

# 创建解释器
explainer = shap.Explainer(model, X_test)
shap.summary_plot(shap_values2, X_test)
import numpy as np
abs_shap_values = np.abs(shap_values2.values)
mean_abs_shap_values = np.mean(abs_shap_values, axis=0)
sorted_indices = np.argsort(mean_abs_shap_values)
sorted_features = X_test.columns[sorted_indices]
sorted_mean_abs_shap_values = mean_abs_shap_values[sorted_indices]
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_features)), sorted_mean_abs_shap_values, align='center')
plt.yticks(range(len(sorted_features)), sorted_features)
plt.xlabel('SHAP平均绝对值')
plt.ylabel('Features')
plt.title('SHAP平均影响排序图')
plt.tight_layout()
plt.show()

y_filtered_df = pd.DataFrame(y_filtered, columns=['Death'])  # 可能需要调整列名以符合你的数据
sorted_features = sorted_features[::-1]
X_filtered_sorted = X_filtered[sorted_features]
X_sorted=X[sorted_features]
y_sorted=y
# 将排序后的特征数据框与目标变量数据框合并
filtered_data_sorted = pd.concat([X_filtered_sorted, y_filtered_df], axis=1)
filtered_data = pd.concat([X_filtered, y_filtered_df], axis=1)
filtered_data_sorted.to_csv('filtered_data_xgb_notime.csv', index=False)  # 设置 index=False 以避免保存行索引

data_sorted = pd.concat([X_sorted, y], axis=1)

data_sorted.to_csv('sorted_data_xgb_notime.csv', index=False)  # 设置 index=False 以避免保存行索引

