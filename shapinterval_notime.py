# -*- coding: utf-8 -*-
"""
这个代码里有且仅有Spearman相关系数矩阵最终呈现在论文中
"""


from sklearn.ensemble import RandomForestClassifier  
import pandas as pd  
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler  
from sklearn.compose import ColumnTransformer  
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV  
import graphviz 
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import accuracy_score 
import pandas as pd  
import numpy as np  
from collections import Counter  
from sklearn.preprocessing import LabelEncoder  
from sklearn.impute import SimpleImputer  # 用于处理缺失值  
from sklearn import model_selection 
from sklearn import tree
from sklearn.tree import plot_tree  
import seaborn as sns  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
from sklearn.preprocessing import StandardScaler  
import numpy as np  
from sklearn.datasets import load_iris  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import KFold, cross_val_score  


credit = pd.read_csv("heatmap.csv",encoding='gbk')

df=credit
# 计算相关性矩阵
corr_pearson = df.corr(method='pearson').iloc[0:20, :20]  # 排除目标列
corr_spearman = df.corr(method='spearman').iloc[0:20, :20]
corr_kendall = df.corr(method='kendall').iloc[0:20, :20]
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


import colorsys

def desaturate_color(rgb, saturation_factor=0.6, value_factor=1.0):
    """降低颜色饱和度并保持色相"""
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = max(0, min(s * saturation_factor, 1.0))  # 降低饱和度
    v = max(0, min(v * value_factor, 1.0))       # 保持明度
    return colorsys.hsv_to_rgb(h, s, v)

# 我设置的颜色映射
original_colors = [
    (1.0, 0.792, 0.113),  # ECFF43
    (153/255, 246/255, 4/255),    # 99F604
    (15/255, 110/255, 220/255)    # 0F6EDC
]

# 灰度增强参数配置
adjusted_colors = [desaturate_color(c, saturation_factor=0.7, value_factor=0.95)
                   for c in original_colors]

# 创建灰度增强的色带
cmap_grayish = mcolors.LinearSegmentedColormap.from_list(
    'grayish_gradient',
    adjusted_colors,
    N=256
)
cmap_custom = mcolors.LinearSegmentedColormap.from_list(
    'custom_gradient', adjusted_colors, N=256)

import pandas as pd
# 绘制热力图spearman
plt.figure(figsize=(10, 8))
threshold = 0.8
high_corr_coords = [(i, j) for i in range(len(corr_spearman)) for j in range(len(corr_spearman))
                    if abs(corr_spearman.iloc[i, j]) > threshold and i != j]

# 绘制热图但不显示注释
plt.figure(figsize=(10, 8))
sns.heatmap(corr_pearson, annot=False, cmap=cmap_custom, center=0)
plt.title('Spearman Correlation Heatmap')

# 在热图上标注特定坐标
for i, j in high_corr_coords:
    # 获取坐标对应的值（用于标注，但这里我们不显示它）
    value = corr_spearman.iloc[i, j]
    # 计算文本位置（这里简单地在单元格中心位置标注，但可能需要调整以避免重叠）
    text_x = j + 0.5
    text_y = i + 0.5
    # 标注坐标（但不显示具体值，只显示一个标记，比如'*'）
    plt.text(text_x, text_y, '*', fontsize=12, ha='center', va='center', color='red')

# 显示图像
plt.show()


# 绘制热力图pearson

plt.figure(figsize=(10, 8))
threshold = 0.8
high_corr_coords = [(i, j) for i in range(len(corr_pearson)) for j in range(len(corr_pearson))
                    if abs(corr_pearson.iloc[i, j]) > threshold and i != j]

# 绘制热图但不显示注释
plt.figure(figsize=(10, 8))
sns.heatmap(corr_pearson, annot=False, cmap='coolwarm', center=0)
plt.title('Pearson Correlation Heatmap')

# 在热图上标注特定坐标
for i, j in high_corr_coords:
    # 获取坐标对应的值（用于标注，但这里我们不显示它）
    value = corr_pearson.iloc[i, j]
    # 计算文本位置（这里简单地在单元格中心位置标注，但可能需要调整以避免重叠）
    text_x = j + 0.5
    text_y = i + 0.5
    # 标注坐标（但不显示具体值，只显示一个标记，比如'*'）
    plt.text(text_x, text_y, '*', fontsize=12, ha='center', va='center', color='red')

# 显示图像
plt.show()


# Kendall
plt.figure(figsize=(10, 8))
threshold = 0.8
high_corr_coords = [(i, j) for i in range(len(corr_kendall)) for j in range(len(corr_kendall))
                    if abs(corr_kendall.iloc[i, j]) > threshold]

# 绘制热图但不显示注释
plt.figure(figsize=(10, 8))
sns.heatmap(corr_kendall, annot=False, cmap='coolwarm', center=0)
plt.title('Kendall Correlation Heatmap')

# 在热图上标注特定坐标
for i, j in high_corr_coords:
    # 获取坐标对应的值（用于标注，但这里我们不显示它）
    value = corr_kendall.iloc[i, j]
    # 计算文本位置（这里简单地在单元格中心位置标注，但可能需要调整以避免重叠）
    text_x = j + 0.5
    text_y = i + 0.5
    # 标注坐标（但不显示具体值，只显示一个标记，比如'*'）
    plt.text(text_x, text_y, '*', fontsize=12, ha='center', va='center', color='red')

# 显示图像
plt.show()

# 使用SHAP解释模型
# 这里我们使用随机森林分类器作为示例模型
X = credit.iloc[0:1611, :-5]  # 特征  
y = credit.iloc[0:1611, -1]   # 目标变量  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# 假设model已经训练好，X_train是训练数据
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)  # 获取SHAP值
corr_matrix = X_train.corr()  # 假设X_train是DataFrame格式
 
# 使用seaborn绘制相关性热图
plt.figure(figsize=(10, 8))
threshold = 0.8
high_corr_coords = [(i, j) for i in range(len(corr_matrix)) for j in range(len(corr_matrix))
                    if abs(corr_matrix.iloc[i, j]) > threshold and i != j]

# 绘制热图但不显示注释
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('SHAP Correlation Heatmap')

# 在热图上标注特定坐标
for i, j in high_corr_coords:
    # 获取坐标对应的值（用于标注，但这里我们不显示它）
    value = corr_matrix.iloc[i, j]
    # 计算文本位置（这里简单地在单元格中心位置标注，但可能需要调整以避免重叠）
    text_x = j + 0.5
    text_y = i + 0.5
    # 标注坐标（但不显示具体值，只显示一个标记，比如'*'）
    plt.text(text_x, text_y, '*', fontsize=12, ha='center', va='center', color='red')

# 显示图像
plt.show()


# 获取特征的重要性（取绝对值并求平均）
feature_importance = np.abs(shap_values).mean(axis=0).sum(axis=1)  # 对于多类分类，需要额外处理shap_values的形状
 

 
# 为了确保特征重要性从大到小排序，我们先对 feature_importance 和 X_train.columns 进行排序
sorted_indices = np.argsort(-feature_importance)  # 使用-feature_importance进行降序排序
sorted_importance = feature_importance[sorted_indices]
sorted_feature_names = X_train.columns[sorted_indices]
 
# 绘制条形图，此时使用plt.bar而非plt.barh
plt.figure(figsize=(10, 6))
plt.bar(sorted_feature_names, sorted_importance, color='skyblue')  # 设定天蓝色条，特征名为X轴标签
plt.xlabel('Feature')  # 原Y轴标签现变为X轴标签
plt.ylabel('Feature Importance')  # 原X轴标签现变为Y轴标签
plt.title('SHAP Feature Importance')
# 不再需要反转轴，因为已经是按X轴（原Y轴翻转后的位置）排列
plt.xticks(rotation=45, ha='right')  # 可选：旋转X轴标签，避免文字重叠
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()