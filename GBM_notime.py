# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:33:46 2024

@author: Administrator
"""
#res=10   91     结果不好
#train=17    91   特征太多 
#resampled=9/19 good 93
#resampled2=
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:37:49 2024

@author: Administrator
"""
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import lightgbm as lgb  
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.metrics import accuracy_score  
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
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import RandomOverSampler  # 注意这里使用了imblearn库
from imblearn.under_sampling import RandomUnderSampler  # 注意这里使用了imblearn库
from imblearn.over_sampling import SMOTE
import logging
logging.getLogger('lightgbm').setLevel(logging.ERROR)  # 只显示错误信息，隐藏警告和调试信息

credit2=pd.read_csv("filtered_data_GBM_notime.csv",encoding='gbk')
credit = pd.read_csv("sorteddataset_GBM_notime.csv",encoding='gbk')


import warnings
warnings.filterwarnings('ignore')
X = credit.iloc[0:1610, :1]  # 特征  
y = credit.iloc[0:1610, -1]   # 目标变量  
  
# 划分训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  
X_train=credit2.iloc[0:, :1]
y_train=credit2.iloc[0:, -1]
ros = RandomOverSampler(random_state=42)
ros2 = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
X_resampled2, y_resampled2 = ros2.fit_resample(X_train, y_train)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
# 创建LightGBM分类器  
lgb_clf = lgb.LGBMClassifier()  
  
# 定义参数网格  
param_grid = {  
    'n_estimators':[50,500,100],
    'learning_rate':[0.001,0.005,0.01,0.015,0.02,0.03,0.025]
}  
cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=7)
# 使用GridSearchCV进行网格搜索  
grid_search = GridSearchCV(estimator=lgb_clf, param_grid=param_grid, cv=3,scoring='accuracy',verbose=1,n_jobs=-1)  
  
# 拟合模型  
grid_search.fit(X_train, y_train)  
  
# 输出最佳参数和最佳得分  
print("Best parameters found: ", grid_search.best_params_)  
from sklearn.model_selection import cross_val_predict, StratifiedKFold
print("Best cross-validation accuracy: {:.4f}".format(grid_search.best_score_))  

  
# 使用最佳参数进行预测  
best_model = grid_search.best_estimator_  
y_pred = best_model.predict(X_test)  
  
# 计算测试集上的准确率  
test_accuracy = accuracy_score(y_test, y_pred)  
print("Test set accuracy: {:.4f}".format(test_accuracy))
names=['Alive','Dead']  
# 输出分类报告  
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)  
print("Confusion Matrix:\n", cm)  
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay  
 
# 绘制混淆矩阵  
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)  
disp.plot(cmap=plt.cm.Blues)  
plt.title('Confusion Matrix')

print(classification_report(y_test, y_pred))
