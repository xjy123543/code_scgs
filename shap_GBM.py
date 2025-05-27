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
import xgboost as xgb
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
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集，这里使用Iris数据集作为示例
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb 
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler  # 注意这里使用了imblearn库
from imblearn.under_sampling import RandomUnderSampler  # 注意这里使用了imblearn库
import numpy as np  
from sklearn.datasets import load_iris  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import KFold, cross_val_score 
import pandas as pd  
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler  
from sklearn.compose import ColumnTransformer  
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV  
import graphviz 
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


credit = pd.read_csv("sorteddataset_Random.csv",encoding='gbk')

# 计算信息熵的函数  
def entropy(y):  
    counter = Counter(y)  
    probabilities = [count / len(y) for count in counter.values()]  
    return -sum(p * np.log2(p) for p in probabilities if p > 0)  
  
# 计算条件熵的函数  
def conditional_entropy(X, y):  
    unique_X = np.unique(X)  
    weighted_entropy = 0.0  
    for x in unique_X:  
        subset = y[X == x]  
        if len(subset) > 0:  # 确保子集非空以避免计算错误  
            weighted_entropy += (len(subset) / len(y)) * entropy(subset)  
    return weighted_entropy  
  
# 计算信息增益的函数  
def information_gain(X, y):  
    base_entropy = entropy(y)  
    cond_entropy = conditional_entropy(X, y)  
    return base_entropy - cond_entropy  
  
# 读取数据集（假设是CSV文件）  
try:  
    df = credit  
except FileNotFoundError:  
    print("文件未找到，请检查文件路径。")  
    exit(1)  
except pd.errors.EmptyDataError:  
    print("文件为空，请检查文件内容。")  
    exit(1)  
except pd.errors.ParserError:  
    print("文件解析错误，请检查文件格式。")  
    exit(1)  
except Exception as e:  
    print(f"读取文件时发生错误: {e}")  
    exit(1)     
feature_column_names = ['age',	'Local','Neck',	'T','N','gender_1.0','gender_2.0','location_1.0','location_2.0','location_3.0','location_4.0',
        'location_5.0',	'location_6.0',	'location_7.0',	'location_8.0',	'location_9.0',	'location_10.0'	,'type_10',
        'type_11','type_12'	,'type_13',	'type_14',	'type_15',	'type_1a',	'type_1a', 	'type_1b',
        'type_1c',	'type_2','type_3'	,'type_4','type_5','type_6','type_7','type_8','type_9',
        'Radiotherapy_0.0','Radiotherapy_1.0','before_after_-1.0','before_after_0.0','before_after_1.0','Chemotherapy_0.0','Chemotherapy_1.0'
 ] # 替换为你的特征列名  
label_column_name = 'Death'  # 替换为你的标签列名  
for feature_column_name in feature_column_names:     
    if feature_column_name not in df.columns or label_column_name not in df.columns:  
        print(f"列名 {feature_column_name} 或 {label_column_name} 不存在于数据集中。")  
        
      
    # 检查特征列是否包含字符串，并进行必要的预处理  
    if df[feature_column_name].dtype == 'object':  
        le = LabelEncoder()  
        try:  
            df[feature_column_name] = le.fit_transform(df[feature_column_name])  
        except ValueError as e:  
            print(f"标签编码时发生错误: {e}")  
            exit(1)  
      
    # 处理缺失值  
    imputer = SimpleImputer(strategy='most_frequent')  # 或者使用 'mean', 'median', 'constant' 等策略  
    df[feature_column_name] = imputer.fit_transform(df[[feature_column_name]].astype(float))  # 转换为float以处理可能的数字  
    df[label_column_name] = df[label_column_name].fillna(df[label_column_name].mode()[0])  # 处理标签列的缺失值  
      
    # 确保标签列是二分类的（如果不是，需要额外的处理）  
    if not set(df[label_column_name]).issubset({0, 1}):  
        print(f"标签列 {label_column_name} 必须包含0和1作为类别标签。")  
        exit(1)  
      
    # 将标签列转换为整型（如果还不是）  
    df[label_column_name] = df[label_column_name].astype(int)  
      
    # 计算信息增益  
    try:  
        ig = information_gain(df[feature_column_name], df[label_column_name])  
        print(f"{feature_column_name}的信息增益: {ig}")  
    except Exception as e:  
        print(f"计算信息增益时发生错误: {e}")  
  
        exit(1)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler  # 注意这里使用了imblearn库
from imblearn.under_sampling import RandomUnderSampler  # 注意这里使用了imblearn库
from imblearn.over_sampling import SMOTE

# 划分训练集和测试集
X = credit.iloc[0:1610, :39]  # 特征
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
model = lgb.LGBMClassifier(max_depth=4)  
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model, X_train)
shap_values = explainer.shap_values(X_train)
# 创建解释器
explainer = shap.Explainer(model, X_train)

# 计算 SHAP 值
shap_values = explainer(X_train)

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
thresholds = mean_shap + 8* std_shap

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
model2 = lgb.LGBMClassifier()
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

# 将排序后的特征数据框与目标变量数据框合并
filtered_data_sorted = pd.concat([X_filtered_sorted, y_filtered_df], axis=1)
filtered_data = pd.concat([X_filtered, y_filtered_df], axis=1)
filtered_data_sorted.to_csv('filtered_data_GBMq.csv', index=False)  # 设置 index=False 以避免保存行索引


