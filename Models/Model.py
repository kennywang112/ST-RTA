import os

current_dir = os.getcwd()
analyze_path = os.path.join(current_dir, "utils")

os.chdir(analyze_path)

import ast
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
import geopandas as gpd

import torch
from torch import nn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print('read data')
dataA1 = pd.read_csv('../ComputedData/Accident/DataA1_with_MYP.csv')
dataA2 = pd.read_csv('../ComputedData/Accident/DataA2_with_MYP.csv')

filtered_A2 = dataA2[dataA2['當事者順位'] == 1]
filtered_A1 = dataA1[dataA1['當事者順位'] == 1]

filtered_A1['source'] = 'A1'
filtered_A2['source'] = 'A2'
filtered_A1['num_accidents'] = 1 
filtered_A2['num_accidents'] = 1
combined_data = pd.concat([filtered_A1, filtered_A2], ignore_index=True)

taiwan = gpd.read_file('../Data/OFiles_9e222fea-bafb-4436-9b17-10921abc6ef2/TOWN_MOI_1140318.shp')
taiwan = taiwan[(~taiwan['TOWNNAME'].isin(['旗津區', '頭城鎮', '蘭嶼鄉', '綠島鄉', '琉球鄉'])) & 
                (~taiwan['COUNTYNAME'].isin(['金門縣', '連江縣', '澎湖縣']))]

TM2 = 3826
hex_grid_raw = pd.read_csv('../ComputedData/Grid/hex_grid.csv')
hex_grid_raw['geometry'] = hex_grid_raw['geometry'].apply(wkt.loads)
hex_grid = gpd.GeoDataFrame(hex_grid_raw, geometry='geometry').set_crs(TM2, allow_override=True)

grid_gi_df = pd.read_csv('../ComputedData/Grid/grid_gi.csv')
grid_gi_df['accident_indices'] = grid_gi_df['accident_indices'].apply(ast.literal_eval)
grid_gi_df['geometry'] = grid_gi_df['geometry'].apply(wkt.loads)
grid_gi  = gpd.GeoDataFrame(grid_gi_df, geometry='geometry').set_crs(TM2, allow_override=True)

taiwan_tm2 = taiwan.to_crs(TM2)

print('join county to grid')

taiwan_cnty = taiwan_tm2[['COUNTYNAME','geometry']].dissolve(by='COUNTYNAME')
taiwan_cnty['geometry'] = taiwan_cnty.buffer(0)
taiwan_cnty = taiwan_cnty.reset_index()

pts = hex_grid.copy()
pts['geometry'] = pts.geometry.centroid

county_join = gpd.sjoin(
    pts[['geometry']], taiwan_cnty, how='left', predicate='within'
)[['COUNTYNAME']]

grid_gi['COUNTYNAME'] = county_join['COUNTYNAME']
county_join.head()

select_group = [
    # 氣候暫不討論
    # '天候名稱', '光線名稱',

    # 道路問題
    '路面狀況-路面鋪裝名稱', '路面狀況-路面狀態名稱', '路面狀況-路面缺陷名稱',
    '道路障礙-障礙物名稱', '道路障礙-視距品質名稱', '道路障礙-視距名稱',

    # 號誌
    '號誌-號誌種類名稱', '號誌-號誌動作名稱',

    # 車道劃分
    '車道劃分設施-分道設施-快車道或一般車道間名稱',
    '車道劃分設施-分道設施-快慢車道間名稱', '車道劃分設施-分道設施-路面邊線名稱',

    # 大類別
    # '肇因研判大類別名稱-主要', '肇因研判大類別名稱-個別', # 聚焦道路類型
    # '當事者區分-類別-大類別名稱-車種', # 聚焦道路類型
    # '當事者行動狀態大類別名稱', # 聚焦道路類型
    '車輛撞擊部位大類別名稱-最初', #'車輛撞擊部位大類別名稱-其他',
    '事故類型及型態大類別名稱', '車道劃分設施-分向設施大類別名稱',
    '事故位置大類別名稱', '道路型態大類別名稱',
    
    # 子類別
    # '肇因研判子類別名稱-主要', '肇因研判子類別名稱-個別', # 聚焦道路類型
    # '當事者區分-類別-子類別名稱-車種', # 聚焦道路類型
    # '當事者行動狀態子類別名稱', # 聚焦道路類型
    # '車輛撞擊部位子類別名稱-最初', '車輛撞擊部位子類別名稱-其他', # 道路類型很大程度影響撞擊部位，所以不考慮
    # '事故類型及型態子類別名稱', '車道劃分設施-分向設施子類別名稱', 
    # '事故位置子類別名稱', '道路型態子類別名稱',

    # 其他
    # '當事者屬-性-別名稱', '當事者事故發生時年齡', 
    '速限-第1當事者', '道路類別-第1當事者-名稱',
    # '保護裝備名稱', '行動電話或電腦或其他相類功能裝置名稱', '肇事逃逸類別名稱-是否肇逃',

    # 設施
    'youbike_100m_count', 'mrt_100m_count', 'parkinglot_100m_count',

    # A1 or A2
    # 'source',
    ]

print('read full feature')
all_features_df = pd.read_csv("../ComputedData/ForModel/all_features.csv")
grid_filter = grid_gi[grid_gi['accident_indices'].str.len() > 0]

# with county town
# 原始資料index並非從1開始所以需reset
new_grid = pd.concat([grid_filter.reset_index(drop=True)[['hotspot', 'COUNTYNAME']], all_features_df], axis=1)
county_dummies = pd.get_dummies(new_grid['COUNTYNAME'], prefix='county')
new_grid_encoded = pd.concat([new_grid.drop(['COUNTYNAME'], axis=1), county_dummies], axis=1)

# binary hotspot
new_grid_encoded['hotspot'] = new_grid_encoded['hotspot'].apply(lambda x: 'Hotspot' if 'Hotspot' in str(x) else 'Not Hotspot')

le = LabelEncoder()
y = le.fit_transform(new_grid_encoded['hotspot'])
X = new_grid_encoded.drop(columns=['hotspot'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
y_train = pd.Series(y_train, index=X_train.index)
y_test  = pd.Series(y_test,  index=X_test.index)

# with undersampling
cls_counts = y_test.value_counts()
min_count = cls_counts.min()
rus_test = RandomUnderSampler(
    sampling_strategy={int(c): int(min_count) for c in cls_counts.index},
    random_state=42
)
X_resampled_test, y_resampled_test = rus_test.fit_resample(X_test, y_test)

print('start LR and RF training')

lr = LogisticRegression(
        class_weight='balanced', max_iter=1000, 
        random_state=42, 
        multi_class='multinomial'
    )
rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=1,
        class_weight='balanced', n_jobs=-1, random_state=42,
    )

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in [('Logistic', lr), ('RandomForest', rf)]:
    scores = cross_val_score(clf, X_train, y_train, cv=cv, n_jobs=-1,
                            scoring='roc_auc_ovr_weighted',
                            # scoring='roc_auc'
                             )
    print(f'{name} CV ROC AUC: {scores.mean():.3f} ± {scores.std():.3f}')

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

proba_test_lr = lr.predict_proba(X_resampled_test)
proba_test_rf = rf.predict_proba(X_resampled_test)
y_pred_lr = np.argmax(proba_test_lr, axis=1)
y_pred_rf = np.argmax(proba_test_rf, axis=1)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print('start neural network training')

X_train_t = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
X_test_t  = torch.from_numpy(np.asarray(X_resampled_test,  dtype=np.float32))
y_test_t  = torch.from_numpy(np.asarray(y_resampled_test,  dtype=np.int64))

INPUT_DIM = X_resampled_test.shape[1]
NUM_CLASSES = int(len(set(y)))  # 類別 0/1

class BinaryMLP(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, num_classes=NUM_CLASSES, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, num_classes)  # logits
        )
    def forward(self, x):
        return self.net(x)

model = BinaryMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

model.train()
epochs = 30
batch_size = 256
for epoch in range(epochs):
    # Train
    perm = torch.randperm(X_train_t.size(0)) # 隨機排列訓練集中每一筆資料的索引
    epoch_loss = 0.0
    for i in range(0, X_train_t.size(0), batch_size):
        idx = perm[i:i+batch_size]
        xb = X_train_t[idx].to(device)
        yb = y_train_t[idx].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)

    # Val
    model.eval()
    with torch.no_grad():
        logits_val = model(X_test_t.to(device))
        # 回傳最大值所在位置
        preds_val = logits_val.argmax(dim=-1)
        acc_val = (preds_val.cpu() == y_test_t).float().mean().item()
    model.train()

    if device.type == "mps":
        torch.mps.synchronize()

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {epoch_loss / X_train_t.size(0):.4f} | "
          f"Val Acc: {acc_val:.4f}")

models = {
    'RandomForest': y_pred_rf,
    'LogisticRegression': y_pred_lr,
    'NeuralNetwork': preds_val.cpu().numpy()
}

save_dir = "../ComputedData/ModelPerformance"
os.makedirs(save_dir, exist_ok=True)
with open(f'{save_dir}/models.pkl', 'wb') as f:
    pickle.dump(models, f)