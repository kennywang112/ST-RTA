import re
import torch
import numpy as np
import pandas as pd
from config import for_poly
from itertools import combinations
from collections import defaultdict
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score, accuracy_score, f1_score, recall_score, precision_score

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_importance(model, df, specific_col=None):
    if model.__class__.__name__ == 'LogisticRegression':
        importances = model.coef_[0]
    else:
        importances = model.feature_importances_

    feature_names = df.columns

    if specific_col:
        sel_idx = [i for i, name in enumerate(feature_names) if specific_col in name]
        indices = np.argsort(importances[sel_idx])[::-1]
        indices = [sel_idx[i] for i in indices] # 對應回原始 index
    else:
        indices = np.argsort(importances)[::-1]

    # 以縣市為單位進行分組
    importance_ungrouped = {}
    for i in indices:
        importance_ungrouped[feature_names[i]] = [importances[i], np.exp(importances[i])]

    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df['main_feature'] = fi_df['feature'].str.split('_').str[0]
    grouped = fi_df.groupby('main_feature', as_index=False)['importance'].sum()
    grouped['exp'] = np.exp(grouped['importance'])
    grouped = grouped.sort_values('importance', ascending=False)

    return importance_ungrouped, grouped

def extract_features(
        grid, combined_data, select_group, rows
        ):

    indices = grid['accident_indices'].iloc[rows] # return list of original data index
    sample = combined_data.iloc[indices]
    sample = sample[select_group]

    cat_cols = sample.select_dtypes(include='object').columns
    num_cols = sample.select_dtypes(include='number').columns

    cat_features = []
    num_features = []
    # for categorical features
    if len(cat_cols) > 0:
        for col in cat_cols:
            vc = sample[col].value_counts(normalize=True)
            vc.index = [f"{col}_{v}" for v in vc.index]
            cat_features.append(vc)
        cat_features = pd.concat(cat_features)
    else:
        cat_features = pd.Series(dtype='float64')
    # for numerical features
    if len(num_cols) > 0:
        num_features = sample[num_cols].mean()
        num_features.index = [f"{col}_mean" for col in num_features.index]
    else:
        num_features = pd.Series(dtype='float64')

    all_features = pd.concat([cat_features, num_features])
    all_features_df = all_features.to_frame().T

    return all_features_df

def model_preprocess(grid_filter, all_features_df):
    # with county town
    # 原始資料index並非從1開始所以需reset
    new_grid = pd.concat([grid_filter[['COUNTYNAME']], all_features_df], axis=1)
    county_dummies = pd.get_dummies(new_grid['COUNTYNAME'], prefix='county')
    new_grid_encoded = pd.concat([new_grid.drop(['COUNTYNAME'], axis=1), county_dummies], axis=1)

    # binary hotspot
    new_grid_encoded['hotspot'] = new_grid_encoded['hotspot'].apply(lambda x: 'Hotspot' if 'Hotspot' in str(x) else 'Not Hotspot')
    le = LabelEncoder()
    # y = le.fit_transform(new_grid_encoded['hotspot'])
    y = new_grid_encoded['hotspot'].map({'Not Hotspot': 0, 'Hotspot': 1}).values
    X = new_grid_encoded.drop(columns=['hotspot'])
    le.classes_ = ['Not Hotspot', 'Hotspot']

    # interaction
    from utils_model import get_interaction
    X_interaction = get_interaction(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_interaction, y, test_size=0.2, stratify=y, random_state=42
    )
    y_train = pd.Series(y_train, index=X_train.index)
    y_test  = pd.Series(y_test,  index=X_test.index)

    # undersampling
    cls_counts = y_test.value_counts()
    min_count = cls_counts.min()
    rus_test = RandomUnderSampler(
        sampling_strategy={int(c): int(min_count) for c in cls_counts.index},
        random_state=42
    )
    X_resampled_test, y_resampled_test = rus_test.fit_resample(X_test, y_test)

    return X_train, X_test, y_train, y_test, X_resampled_test, y_resampled_test, le

def get_interaction(X):

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

    return X

def build_groups_from_prefix(columns, sep="_"):
    """
    最基礎的分開方式，原本適用於不考慮交互作用的模型
    """
    groups = {}
    for c in columns:
        prefix = c.split(sep, 1)[0]
        groups.setdefault(prefix, []).append(c)
    return groups

def build_groups_with_interactions(columns, base_sep="_", inter_pattern=r"\s*x\s*"):
    """
    考慮交互作用，但顯示還是以個別欄位為單位
    """
    def main_prefix(s: str) -> str:
        return s.split(base_sep, 1)[0]
    groups = defaultdict(list)

    for col in columns:
        # 先照原本規則，把欄位放到自己的主前綴群組
        base = main_prefix(col)
        groups[base].append(col)

        # 再處理交互作用：以 x（不分大小寫、容許空白）切開
        parts = re.split(inter_pattern, col, flags=re.IGNORECASE)
        if len(parts) > 1:
            for p in parts:
                g = main_prefix(p)
                if col not in groups[g]:
                    groups[g].append(col)

    return dict(groups)

def build_pair_interaction_groups(columns, base_sep="_", inter_pattern=r"\s*x\s*"):
    """
    單位是兩個不同欄位的交互作用，然後再細分裡面的值
    """
    def main_prefix(s: str) -> str:
        return s.split(base_sep, 1)[0]
    pair_groups = defaultdict(list)

    for col in columns:
        parts = re.split(inter_pattern, col, flags=re.IGNORECASE)
        if len(parts) <= 1:
            continue  # 非交互作用欄位略過

        # 抽取每個部分的主前綴（去重以免同一前綴重複）
        prefixes = [main_prefix(p) for p in parts]
        unique_prefixes = list(dict.fromkeys(prefixes))  # 保留順序的去重

        a, b = unique_prefixes
        if a > b: a, b = b, a

        key = f"{a} x {b}"
        pair_groups[key].append(col)

    return dict(pair_groups)

def PI_ML(
        model, X_df, y, groups=None, n_repeats=5, random_state=42
        ):

    X_arr = X_df.to_numpy(copy=True)
    name_to_idx = {c: i for i, c in enumerate(X_df.columns)}
    if groups is None:
        idx_groups = {c: [i] for i, c in enumerate(X_df.columns)}
    else:
        idx_groups = {g: [name_to_idx[c] for c in cols] for g, cols in groups.items()}

    def get_ap(proba):
        # 二元：取正類；多類：取第二欄 (假設 label encoder 對應)
        if proba.ndim == 2 and proba.shape[1] > 1:
            target_scores = proba[:, 1]
        else:
            target_scores = proba
        return average_precision_score(y, target_scores)

    rng = np.random.RandomState(random_state)
    base = get_ap(model.predict_proba(X_arr))
    n_samples = X_arr.shape[0]
    X_work = X_arr.copy()

    rows = []
    for gname, cols in idx_groups.items():
        original = X_work[:, cols].copy()
        losses = []
        for _ in range(n_repeats):
            perm_idx = rng.permutation(n_samples)
            X_work[:, cols] = original[perm_idx, :]
            score = get_ap(model.predict_proba(X_work))
            losses.append(base - score)
            X_work[:, cols] = original  # restore
        rows.append((gname, float(np.mean(losses)), float(np.std(losses))))

    out = pd.DataFrame(rows, columns=["group", "importance", "std"])\
             .sort_values("importance", ascending=False)\
             .reset_index(drop=True)
    return base, out

def PI_NN(model, X_df, y, groups=None, n_repeats=5, random_state=42):

    X_arr = X_df.to_numpy(copy=True)
    name_to_idx = {c: i for i, c in enumerate(X_df.columns)}
    if groups is None:
        idx_groups = {c: [i] for i, c in enumerate(X_df.columns)}
    else:
        idx_groups = {g: [name_to_idx[c] for c in cols] for g, cols in groups.items()}

    def get_ap(model, X_work):

        Xt = torch.from_numpy(np.asarray(X_work,  dtype=np.float32))
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        logits_val = model(Xt.to(device))
        # 回傳最大值所在位置
        preds_val = logits_val.argmax(dim=-1)
        acc_val = (preds_val.cpu() == yt).float().mean().item()
        return acc_val

    rng = np.random.RandomState(random_state)
    base = get_ap(model, X_arr)
    n_samples = X_arr.shape[0]
    X_work = X_arr.copy()

    rows = []
    for gname, cols in idx_groups.items():
        original = X_work[:, cols].copy()
        losses = []
        for _ in range(n_repeats):
            perm_idx = rng.permutation(n_samples)
            X_work[:, cols] = original[perm_idx, :]
            score = get_ap(model, X_work)
            losses.append(base - score)
            X_work[:, cols] = original  # restore
        rows.append((gname, float(np.mean(losses)), float(np.std(losses))))

    out = pd.DataFrame(rows, columns=["group", "importance", "std"])\
             .sort_values("importance", ascending=False)\
             .reset_index(drop=True)
    return base, out

def hitrate_data(resample_X, resample_y, model_y):

    county_cols = [col for col in resample_X.columns if col.startswith('county_')]

    df_hitrate = resample_X.copy()
    df_hitrate['y_true'] = resample_y
    df_hitrate['y_pred'] = model_y

    hitrate = {}
    for col in county_cols:

        mask = df_hitrate[df_hitrate[col] != False]
        tn, fp, fn, tp = confusion_matrix(
            mask['y_true'], mask['y_pred'], labels=[1, 0] # 這裡0是Hotspot
        ).ravel()

        # calculate precision, recall, accuracy, f1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        hitrate[col] = {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1': f1
        }

    hitrate_df = pd.DataFrame.from_dict(hitrate, orient='index', columns=['precision', 'recall', 'accuracy', 'f1']).sort_values('f1', ascending=False)
    hitrate_df['county'] = hitrate_df.index
    hitrate_df['county'] = hitrate_df['county'].str.replace('county_', '')

    return hitrate_df

def print_results(proba_test, classes, y_resampled_test):
    """
    proba_test: 預測的概率
    classes: 類別名稱
    y_resampled_test: 重抽樣後的測試標籤
    """
    le = LabelEncoder()

    y_pred = np.argmax(proba_test, axis=1)

    print("Confusion Matrix")
    print(confusion_matrix(y_resampled_test, y_pred, labels=range(len(classes))))

    print("Classification Report")
    print(classification_report(
        y_resampled_test, y_pred, target_names=classes, digits=3
    ))

    if proba_test.shape[1] == 2:
        # 二元分類
        roc_auc = roc_auc_score(y_resampled_test, proba_test[:, 1])
        print(f'ROC AUC: {roc_auc:.3f}')
        y_test_bin = label_binarize(y_resampled_test, classes=range(len(classes)))
        pr_auc_macro  = average_precision_score(y_test_bin, proba_test[:, 1], average='macro')
        pr_auc_weight = average_precision_score(y_test_bin, proba_test[:, 1], average='weighted')
        print(f'PR  AUC macro: {pr_auc_macro:.3f}')
        print(f'PR  AUC wighted: {pr_auc_weight:.3f}')
    else:
        # 多類分類
        roc_auc = roc_auc_score(y_resampled_test, proba_test, average='weighted', multi_class='ovr')
        print(f'ROC AUC: {roc_auc:.3f}')
        # 多類PR AUC需要 binarize 後用 one-vs-rest，再做 macro/weighted 平均
        y_test_bin = label_binarize(y_resampled_test, classes=range(len(le.classes_)))  # shape [n, n_classes]
        pr_auc_macro  = average_precision_score(y_test_bin, proba_test, average='macro')
        pr_auc_weight = average_precision_score(y_test_bin, proba_test, average='weighted')
        print(f'PR  AUC macro: {pr_auc_macro:.3f}')
        print(f'PR  AUC wighted: {pr_auc_weight:.3f}')

def to_tensors(X_df, y_arr):
    return (torch.from_numpy(np.asarray(X_df, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_arr, dtype=np.int64)))

def eval_loop(model, loader, le):

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

def metrics_bin(y_true, y_pred):
    return {
        'precision': precision_score(y_true, y_pred, pos_label=0),
        'recall':    recall_score(y_true, y_pred, pos_label=0),
        'f1':        f1_score(y_true, y_pred, pos_label=0),
        'accuracy':  accuracy_score(y_true, y_pred),
    }