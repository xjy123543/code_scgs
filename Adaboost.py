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
from sklearn.model_selection import cross_val_predict, StratifiedKFold

credit = pd.read_csv("sorteddataset_Random.csv",encoding='gbk')
credit2=pd.read_csv("filtered_data_DecisionTree.csv",encoding='gbk')

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
        'type_11','type_12'	,'type_13',	'type_14',	'type_15',	'type_1a', 	'type_1b',
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

X = credit.iloc[0:1610, :1]  # 特征  
y = credit.iloc[0:1610, -1]   # 目标变量  

# 划分训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42) 
X_train=credit2.iloc[0:1371, :1]
y_train=credit2.iloc[0:1371, -1] 
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
ros2 = RandomUnderSampler(random_state=42)
X_resampled2, y_resampled2 = ros2.fit_resample(X_train, y_train)
smote=SMOTE(random_state=42)
X_res,y_res=smote.fit_resample(X_train, y_train)
# 设置AdaBoost分类器的基础学习器（弱学习器）  
base_estimator = DecisionTreeClassifier(max_depth=4,class_weight={1:2.19,0:1})  
# 设置参数网格
param_grid = {  
    'learning_rate': [0.07,0.06,0.08,0.09,0.1,0.2],  
    'estimator__max_depth': [11,10,9,4,3,12]
}  
# 使用GridSearchCV进行参数网格搜索  

  # 初始化AdaBoost分类器  
ada_clf = AdaBoostClassifier(estimator=base_estimator,n_estimators=100)   
grid_search = GridSearchCV(estimator=ada_clf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)   
# 拟合模型  
grid_search.fit(X_res, y_res)  
  
# 输出最佳参数  
print(f"Best parameters found: {grid_search.best_params_}")  
  
# 使用最佳模型进行预测  
best_model = grid_search.best_estimator_  
y_pred = best_model.predict(X_test)  
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
import warnings

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=FutureWarning)
