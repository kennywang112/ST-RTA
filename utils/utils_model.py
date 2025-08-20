import re
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import average_precision_score, confusion_matrix
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
        indices = [sel_idx[i] for i in indices]   # 對應回原始 index
    else:
        indices = np.argsort(importances)[::-1]

    print('feature importance')
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

    print('grouped feature importance')
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df['main_feature'] = fi_df['feature'].str.split('_').str[0]
    grouped = fi_df.groupby('main_feature')['importance'].sum().sort_values(ascending=False)

    print(grouped)

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

# build_groups_from_prefix是最基礎的分開方式，原本適用於不考慮交互作用的模型
# build_groups_with_interactions則是考慮交互作用，但顯示還是以個別欄位為單位
# build_pair_interaction_groups的單位是兩個不同欄位的交互作用，然後再細分裡面的值

def build_groups_from_prefix(columns, sep="_"):
    groups = {}
    for c in columns:
        prefix = c.split(sep, 1)[0]
        groups.setdefault(prefix, []).append(c)
    return groups

def build_groups_with_interactions(columns, base_sep="_", inter_pattern=r"\s*x\s*"):
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
