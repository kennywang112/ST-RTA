import pandas as pd
import numpy as np

import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

from utils.utils_tda import linf_centrality_exact

# Origin
# 目前版本拓樸使用all_features進行建立
all_features_df = pd.read_csv("./ComputedDataV4/ForModel/all_features.csv")
cols = all_features_df.columns[all_features_df.columns.str.contains('hotspot')]
all_features_df.drop(columns=cols, inplace=True)

# pca_model = PCA(n_components=5)
# filter_pca = pca_model.fit_transform(all_features_df)
# feature_names = all_features_df.columns
# loadings_df = pd.DataFrame(
#     pca_model.components_.T, 
#     columns=[f'PC{i+1}' for i in range(5)],
#     index=feature_names
# )

# top_features_pc1 = loadings_df['PC1'].abs().sort_values(ascending=False)
# print(top_features_pc1.head(10))

# PCA
print('Start PCA')
pc=5
filter_pca = PCA(pc).fit_transform(all_features_df)

pca = PCA(pc).fit(all_features_df)
ratios = pca.explained_variance_ratio_

# KDE
print('Start KDE')
X = filter_pca
kde = KernelDensity(kernel='gaussian').fit(X)
log_density = kde.score_samples(X)
density = np.exp(log_density)
# rank-normalize
rank = (np.argsort(np.argsort(density)).astype(float) / (len(density)-1))
filter_kde = rank.reshape(-1, 1) 

# Centrality
print('Start Centrality')
filter_centrality = linf_centrality_exact(all_features_df)
filter_full = np.concatenate([filter_centrality, filter_kde, filter_pca], axis=1)

filter_full = pd.DataFrame(filter_full, columns=['centrality', 'kde', 'pc1', 'pc2', 'pc3', 'pc4', 'pc5'])
filter_full.to_csv("./ComputedDataV4/ForModel/filtered_dataV1.csv", index=False)
