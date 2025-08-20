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
from torch.utils.data import TensorDataset, DataLoader
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score, accuracy_score, f1_score, recall_score, precision_score

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

print('read full feature')
all_features_df = pd.read_csv("../ComputedData/ForModel/all_features.csv")
grid_filter = grid_gi[grid_gi['accident_indices'].str.len() > 0]

# Model preprocess
from config import for_poly
# with county town
# 原始資料index並非從1開始所以需reset
new_grid = pd.concat([grid_filter[['hotspot', 'COUNTYNAME']], all_features_df], axis=1)
county_dummies = pd.get_dummies(new_grid['COUNTYNAME'], prefix='county')
new_grid_encoded = pd.concat([new_grid.drop(['COUNTYNAME'], axis=1), county_dummies], axis=1)

# binary hotspot
new_grid_encoded['hotspot'] = new_grid_encoded['hotspot'].apply(lambda x: 'Hotspot' if 'Hotspot' in str(x) else 'Not Hotspot')

le = LabelEncoder()
y = le.fit_transform(new_grid_encoded['hotspot'])
X = new_grid_encoded.drop(columns=['hotspot'])

###
from itertools import combinations
groups = {base: [c for c in X.columns if c.startswith(base)] for base in for_poly}
# 只做不同基底之間的配對
base_pairs = list(combinations(for_poly, 2))

new_cols = {}
for a, b in base_pairs:
    cols_a, cols_b = groups[a], groups[b]
    for ca in cols_a:
        va = X[ca].values
        for cb in cols_b:
            vb = X[cb].values
            prod = va * vb
            # 若這個交互列完全為0就跳過（節省維度）
            if not np.any(prod):
                continue
            name = f"{ca} x {cb}"
            new_cols[name] = prod

if new_cols:
    X_inter = pd.DataFrame(new_cols, index=X.index)
    X = pd.concat([X, X_inter], axis=1)
###

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
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

print("before US")
print(pd.Series(y_test).map(dict(enumerate(le.classes_))).value_counts())
print("after US")
print(pd.Series(y_resampled_test).map(dict(enumerate(le.classes_))).value_counts())

lr = LogisticRegression(
        penalty='elasticnet', solver='saga', l1_ratio=0.5,
        class_weight='balanced', max_iter=1000, 
        random_state=42, 
        multi_class='multinomial',
        n_jobs=-1
    )
rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=1,
        class_weight='balanced', n_jobs=-1, random_state=42,
    )

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in [('Logistic', lr), ('RandomForest', rf)]:
    scores = cross_val_score(clf, X_train, y_train, cv=cv, n_jobs=-1,
                            # scoring='roc_auc_ovr_weighted',
                            scoring='roc_auc'
                             )
    print(f'{name} CV ROC AUC: {scores.mean():.3f} ± {scores.std():.3f}')

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

proba_test_lr = lr.predict_proba(X_resampled_test)
proba_test_rf = rf.predict_proba(X_resampled_test)
y_pred_lr = np.argmax(proba_test_lr, axis=1)
y_pred_rf = np.argmax(proba_test_rf, axis=1)

import joblib
joblib.dump(lr, '../ComputedData/ModelPerformance/lr_model.pkl')
joblib.dump(rf, '../ComputedData/ModelPerformance/rf_model.pkl')

# NN
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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
    
X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

def to_tensors(X_df, y_arr):
    return (torch.from_numpy(np.asarray(X_df, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_arr, dtype=np.int64)))

X_train_t, y_train_t = to_tensors(X_train, y_train)
X_val_t, y_val_t = to_tensors(X_val_nn, y_val_nn)
X_test_t, y_test_t = to_tensors(X_resampled_test, y_resampled_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True, drop_last=False)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=512, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=512, shuffle=False)

model = BinaryMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

best_val = -np.inf
patience = 5
wait = 0
epochs = 20

def eval_loop(loader):
    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_y.append(yb)
    logits_all = torch.cat(all_logits)
    y_all = torch.cat(all_y)
    probs = torch.softmax(logits_all, dim=1).numpy()
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_all, preds)
    f1  = f1_score(y_all, preds, average='binary' if probs.shape[1]==2 else 'weighted')
    recall = recall_score(y_all, preds, average='binary' if probs.shape[1]==2 else 'weighted')
    if probs.shape[1] == 2:
        auc = roc_auc_score(y_all, probs[:,1])
    else:
        auc = roc_auc_score(y_all, probs, multi_class='ovr', average='weighted')

    conf = confusion_matrix(y_all, preds, labels=range(len(le.classes_)))
    report = classification_report(y_all, preds, target_names=le.classes_, digits=3)

    return {'acc': acc, 'f1': f1, 'recall': recall, 'auc': auc, 'conf': conf, 'report': report, 'pred_y': preds}

for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    train_loss = total_loss / len(train_loader.dataset)
    val_metrics = eval_loop(val_loader)
    print(f'Epoch {epoch:02d}/{epochs} | loss {train_loss:.4f} | '
          f'val_acc {val_metrics["acc"]:.3f} | val_f1 {val_metrics["f1"]:.3f} | val_auc {val_metrics["auc"]:.3f}')

    score_for_early = val_metrics["auc"]  # 你也可用 f1
    if score_for_early > best_val:
        best_val = score_for_early
        wait = 0
        # torch.save(model.state_dict(), 'best_model.pt')
    else:
        wait += 1
        if wait >= patience:
            print('Early stopping.')
            break

torch.save(model.state_dict(), '../ComputedData/ModelPerformance/nn_model.pth')