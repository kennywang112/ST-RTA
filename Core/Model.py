import os

current_dir = os.getcwd()
analyze_path = os.path.join(current_dir, "utils")

os.chdir(analyze_path)

import joblib
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils_model import model_preprocess, to_tensors, eval_loop
from utils import read_taiwan_specific

print('read data')
ComputedDataVersion = "V2"
V = '1'
taiwan, grid_filter = read_taiwan_specific(read_grid=True)

# all_featuresV2 為將離群替換為中位數
all_features_df = pd.read_csv(f"../{ComputedDataVersion}/ForModel/all_featuresV{V}.csv")
# 移除高共線
cols = all_features_df.columns[all_features_df.columns.str.contains('事故位置大類別名稱')]
cols2 = all_features_df.columns[all_features_df.columns.str.contains('號誌動作')]
cols3 = all_features_df.columns[all_features_df.columns.str.contains('original_speed')]
all_features_df.drop(columns=cols, inplace=True)
all_features_df.drop(columns=cols2, inplace=True)
all_features_df.drop(columns=cols3, inplace=True)

# Model preprocess
X_train, X_test, y_train, y_test, X_resampled_test, y_resampled_test, le = model_preprocess(grid_filter, all_features_df)

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

joblib.dump(lr, f'../{ComputedDataVersion}/ModelPerformance/lr_model.pkl')
joblib.dump(rf, f'../{ComputedDataVersion}/ModelPerformance/rf_model.pkl')

print('MLP')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
INPUT_DIM = X_resampled_test.shape[1]
NUM_CLASSES = 2

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

    score_for_early = val_metrics["auc"]
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

# Start Permutation
from utils_model import build_groups_with_interactions, PI_ML, PI_NN

groups = build_groups_with_interactions(X_test.columns)

print('lr')
base_lr, perm_lr = PI_ML(lr, X_test, y_test, groups=groups, n_repeats=10)
print('rf') 
base_rf, perm_rf = PI_ML(rf, X_test, y_test, groups=groups, n_repeats=10)
print('nn')
base_nn, perm_nn = PI_NN(model, X_test, y_test, groups=groups, n_repeats=10)

perm_lr.to_csv('../ComputedDataV2/Permutation/perm_lrV1.csv')
perm_rf.to_csv('../ComputedDataV2/Permutation/perm_rfV1.csv')
perm_nn.to_csv('../ComputedDataV2/Permutation/perm_nnV1.csv')