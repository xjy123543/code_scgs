# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:38:55 2024

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 23:33:39 2024

@author: Administrator
"""

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

#
# credit = pd.read_csv("sorteddataset_Decisiontree_notime.csv",encoding='gbk')
# credit2=pd.read_csv("filtered_data_Decisiontree_notime.csv",encoding='gbk')
credit = pd.read_csv("sorteddataset.csv",encoding='gbk')
credit2=pd.read_csv("filtered_data_DecisionTree.csv",encoding='gbk')


from sklearn.model_selection import cross_val_predict, StratifiedKFold
X = credit.iloc[0:1610, :9]  # 特征
y = credit.iloc[0:1610, -1]

# 划分训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42) 
X_train=credit2.iloc[0:1385, :9]
y_train=credit2.iloc[0:1385, -1]
ros = RandomOverSampler(random_state=42)
ros2 = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
X_resampled2, y_resampled2 = ros2.fit_resample(X_train, y_train)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
# 定义随机森林分类器  
rf = RandomForestClassifier()  
  # 定义超参数网格  
param_grid = {  
    'max_depth':[1,4,12,20,50,30],
    'n_estimators':[30,40,20],
    'max_features':[2,4,6,8] ,
    'class_weight':[{1:1,0:2.19}]
}  
kfold=StratifiedKFold(n_splits=3)
# 使用网格搜索进行超参数调优  
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kfold,
                           n_jobs=-1, verbose=3)  
grid_search.fit(X_res, y_res)  

# 输出最优超参数和最优得分  
print("Best parameters found: ", grid_search.best_params_)  
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))  
  
# 在测试集上评估模型  
best_rf = grid_search.best_estimator_  
test_score = best_rf.score(X_test, y_test)  
print("Test set score: {:.2f}".format(test_score))

  
# 在测试集上进行预测  
y_pred = best_rf.predict(X_test)  
names=['Alive','Dead']
# 生成分类报告  
report = classification_report(y_test, y_pred, target_names=names)  
print("Classification Report:\n", report)  
  
# 生成混淆矩阵  
cm = confusion_matrix(y_test, y_pred)  
print("Confusion Matrix:\n", cm)  
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay  
 
#绘制混淆矩阵  
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)  
disp.plot(cmap=plt.cm.Blues)  
plt.title('Confusion Matrix')  
import joblib
import numpy as np
scaler = StandardScaler()
joblib.dump(rf, 'random.pkl')  # 保存模型
joblib.dump(scaler, 'scaler.pkl')     # 保存标准化器（如果需要在GUI中使用相同的标准化）

print("模型训练并保存成功！")
# 保存模型时使用 pickle
import pickle
from sklearn.ensemble import RandomForestClassifier


with open('a.pkl', 'wb') as file:
    pickle.dump(rf, file)
    
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve





# --- 1. 改进 ROC 曲线部分 ---
n_bootstrap = 100 # Bootstrap 次数
fpr_mean = np.linspace(0, 1, 20)  # 固定插值点
tpr_list = []  # 存储插值后的 TPR
auc_list = []  # 存储 AUC

for _ in range(n_bootstrap):
    # 随机采样
    X_resampled, y_resampled = resample(X_test, y_test)
    y_pred_proba_resampled = best_rf.predict_proba(X_resampled)[:, 1]
    fpr, tpr, _ = roc_curve(y_resampled, y_pred_proba_resampled)

    # 插值到固定长度
    tpr_interp = np.interp(fpr_mean, fpr, tpr)  # 插值 tpr
    tpr_list.append(tpr_interp)  # 保存插值结果
    auc_list.append(auc(fpr, tpr))  # 保存 AUC

# 计算均值和置信区间
tpr_mean = np.mean(tpr_list, axis=0)
tpr_std = np.std(tpr_list, axis=0)
mean_auc = np.mean(auc_list)

# 绘制 ROC 曲线
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.plot(fpr_mean, tpr_mean, color='orange', lw=2, label=f'AUC = {mean_auc:.2f}')
plt.fill_between(fpr_mean, tpr_mean - tpr_std, tpr_mean + tpr_std, color='orange', alpha=0.2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Discrimination (ROC Curve)')
plt.legend()

# --- 2. 改进校准曲线部分 ---
n_bins = 5# 分箱数量
fixed_bins = np.linspace(0, 1, n_bins)  # 固定插值点
prob_true_list = []  # 存储插值后的真实概率
prob_pred_list = []  # 存储插值后的预测概率

for _ in range(n_bootstrap):
    X_resampled, y_resampled = resample(X_test, y_test)
    y_pred_proba_resampled = best_rf.predict_proba(X_resampled)[:, 1]
    prob_true, prob_pred = calibration_curve(y_resampled, y_pred_proba_resampled, n_bins=n_bins)
    
    # 插值到固定长度
    prob_true_interp = np.interp(fixed_bins, prob_pred, prob_true)
    prob_true_list.append(prob_true_interp)  # 保存真实概率插值
    prob_pred_list.append(fixed_bins)  # 插值后的固定预测概率

# 计算均值和置信区间
prob_true_mean = np.mean(prob_true_list, axis=0)
prob_true_std = np.std(prob_true_list, axis=0)

# 绘制校准曲线
plt.subplot(1, 3, 2)
plt.plot(fixed_bins, prob_true_mean, marker='o', color='blue', label='Calibration')
plt.fill_between(fixed_bins, prob_true_mean - prob_true_std, prob_true_mean + prob_true_std, color='blue', alpha=0.2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Probability')
plt.title('Calibration Curve')
plt.legend()

# --- 3. 改进决策曲线部分 ---
thresholds = np.linspace(0.01, 0.95, 100)  # 决策阈值范围
net_benefit_list = []

for _ in range(n_bootstrap):
    X_resampled, y_resampled = resample(X_test, y_test)
    y_pred_proba_resampled = best_rf.predict_proba(X_resampled)[:, 1]
    net_benefit = []

    for t in thresholds:
        predicted_positive = y_pred_proba_resampled >= t
        tp = np.sum((predicted_positive == 1) & (y_resampled == 1))
        fp = np.sum((predicted_positive == 1) & (y_resampled == 0))
        benefit = tp - fp * (t / (1 - t))
        net_benefit.append(benefit / len(y_resampled))

    net_benefit_list.append(net_benefit)

# 计算均值和标准差
net_benefit_mean = np.mean(net_benefit_list, axis=0)
net_benefit_std = np.std(net_benefit_list, axis=0)

# 绘制决策曲线
plt.subplot(1, 3, 3)
plt.plot(thresholds, net_benefit_mean, color='red', lw=2, label='Stacking Model')
plt.fill_between(thresholds, net_benefit_mean - net_benefit_std, net_benefit_mean + net_benefit_std, color='red', alpha=0.2)
plt.xlabel('High Risk Threshold')
plt.ylabel('Net Benefit')
plt.title('Clinical Benefit (Decision Curve)')
plt.legend()

plt.tight_layout()
plt.show()
