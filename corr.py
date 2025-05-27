import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 读取数据（假设目标变量在最后一列）
data = pd.read_csv("sorteddataset_Decisiontree_notime.csv", encoding='gbk')
features = data.columns[:20].tolist()  # 假设最后一列是目标变量

# 1. 数据归一化
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)

# 2. 生成所有特征组合
combinations = [(i, j) for i in features for j in features if i != j]

# 3. 预计算相关系数矩阵
corr_matrix = data[features].corr(method='spearman').stack().reset_index()
corr_matrix.columns = ['FeatureA', 'FeatureB', 'Correlation']

# 4. 并行计算联合均值（使用numpy广播加速）
norm_matrix = normalized_data.values
n = len(features)
idx_map = {feat: i for i, feat in enumerate(features)}

# 预计算所有特征对的均值
mean_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            mean_matrix[i, j] = (norm_matrix[:, i] + norm_matrix[:, j]).mean() / 2

# 5. 构建最终结果
result = pd.DataFrame({
    'FeatureA': [pair[0] for pair in combinations],
    'FeatureB': [pair[1] for pair in combinations],
    'Normalized_Mean': [mean_matrix[idx_map[pair[0]], idx_map[pair[1]]] for pair in combinations],
    'Correlation': [corr_matrix.loc[(corr_matrix['FeatureA']==pair[0]) &
                                  (corr_matrix['FeatureB']==pair[1]),
                                  'Correlation'].values[0] for pair in combinations]
})

# 6. 优化内存处理
del norm_matrix, mean_matrix  # 释放大矩阵内存

# 7. 添加解释性排序
result['abs_corr'] = result['Correlation'].abs()
result = result.sort_values(['abs_corr', 'Normalized_Mean'], ascending=[False, False])
result = result.drop('abs_corr', axis=1).reset_index(drop=True)

print(result.head(10))
result.to_csv('feature_interactions.csv',
              index=False,
              encoding='gbk',  # 保持与读取时一致的编码
              float_format='%.4f')  # 控制小数精度
