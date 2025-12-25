import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

def linf_centrality_exact(df, block_size = 2000):
    """
    回傳 shape=(n,1) 的 L∞ centrality(每點到最遠點的距離)
    - metric: "cosine" 或 "euclidean"
    - block_size: 控制記憶體 (block_size * n distances)
    """
    X = df.to_numpy(dtype=float)
    n = X.shape[0]
    # 對每一列作 L2 正規化才能用 cosine 距離
    X = normalize(X, norm="l2", axis=1)

    # 準備結果陣列，初始為無窮小
    max_d = np.full(n, -np.inf, dtype=float)
    order = np.arange(n) # 保留原順序
    
    # 分塊計算 pairwise 距離以控制記憶體
    for start in range(0, n, block_size):
        idx = order[start:start+block_size]
        D_blk = pairwise_distances(X[idx], X, metric='cosine')  # (b, n)
        # 自身距離設為 -inf，避免影響 max
        D_blk[np.arange(D_blk.shape[0]), idx] = -np.inf
        # 針對每個 i（在 idx 中），更新它的全域最遠距離
        max_d[idx] = np.maximum(max_d[idx], D_blk.max(axis=1))

    return max_d.reshape(-1, 1)

def ratio_in_data(data, col='county_city', values='City'):
    """
    choose = 'county_city'
    """
    # 取出要判斷的 Series
    if isinstance(data, pd.DataFrame):
        s = data[col].astype(str)
    else:
        s = pd.Series(data).astype(str)

    if isinstance(values, (list, tuple, set)):
        target = set(map(str, values))
        mask = s.isin(target)
    else:
        mask = (s == str(values))

    return float(mask.mean())

def avg_label(data):
    """
    choose = 'original_speed'
    """
    return sum(data) / len(data) if len(data) > 0 else 0

def most_common_encoded_label(data):
    """
    choose = 'hotspot_facility'
    """
    return Counter(data).most_common(1)[0][0]

def cond_prob_mixed(
        subdf, 
        a_col='hotspot', 
        a_is='Hotspot', 
        b_col='bn_feature', 
        b_rule=">0",
        alpha=0.5, min_den=0, condition="B|A"
    ):
    """
    Return conditional probability:
        - A : 類別欄位
        - B : 可為數值/比例欄位
        - alpha : Laplace smoothing
        - min_den : A 成立的樣本至少要有幾個，否則回 NaN
        - condition="B|A": P(B | A)
        - condition="A|B": P(A | B)

    其他參數說明同前。
    """

    # A: 類別欄位是否落在 a_is 這個集合
    Aset = {a_is} if isinstance(a_is, str) else set(a_is)
    A = subdf[a_col].astype(str)
    mask_A = A.isin(Aset)

    # B: 數值/比例欄位是否滿足 b_rule
    s = pd.to_numeric(subdf[b_col], errors='coerce')
    if callable(b_rule):
        mask_B = b_rule(s)
    else:
        rule = str(b_rule).strip()
        if rule == ">0":
            mask_B = (s > 0)
        elif rule == ">=0":
            mask_B = (s >= 0)
        elif rule.startswith(">="):
            thr = float(rule[2:]); mask_B = (s >= thr)
        elif rule.startswith(">"):
            thr = float(rule[1:]); mask_B = (s > thr)
        elif rule.startswith("<="):
            thr = float(rule[2:]); mask_B = (s <= thr)
        elif rule.startswith("<"):
            thr = float(rule[1:]); mask_B = (s < thr)
        elif rule.startswith("=="):
            thr = float(rule[2:]); mask_B = (s == thr)
        else:
            raise ValueError("Unknown b_rule")

    # 共同分子
    num = (mask_A & mask_B).sum()

    # 選擇分母（誰是條件）
    if condition.upper() == "B|A":
        den = mask_A.sum()    # P(B|A)
    elif condition.upper() == "A|B":
        den = mask_B.sum()    # P(A|B)
    else:
        raise ValueError("condition must be 'B|A' or 'A|B'")

    if den < min_den:
        return float('nan')

    # 對稱的拉普拉斯平滑
    return float((num + alpha) / (den + 2 * alpha))

# conditional probability
def cond_prob_mixed(subdf, a_col, a_is, b_col, b_rule=">0",
                    alpha=0.5, min_den=0, condition="B|A"):
    """
    Return conditional probability:
        - A : 類別欄位
        - B : 可為數值/比例欄位
        - alpha : Laplace smoothing
        - min_den : A 成立的樣本至少要有幾個，否則回 NaN
        - condition="B|A": P(B | A)
        - condition="A|B": P(A | B)
    """

    # A: 類別欄位是否落在 a_is 這個集合
    Aset = {a_is} if isinstance(a_is, str) else set(a_is)
    A = subdf[a_col].astype(str)
    mask_A = A.isin(Aset)

    # B: 數值/比例欄位是否滿足 b_rule
    s = pd.to_numeric(subdf[b_col], errors='coerce')
    if callable(b_rule):
        mask_B = b_rule(s)
    else:
        rule = str(b_rule).strip()
        if rule == ">0":
            mask_B = (s > 0)
        elif rule == ">=0":
            mask_B = (s >= 0)
        elif rule.startswith(">="):
            thr = float(rule[2:]); mask_B = (s >= thr)
        elif rule.startswith(">"):
            thr = float(rule[1:]); mask_B = (s > thr)
        elif rule.startswith("<="):
            thr = float(rule[2:]); mask_B = (s <= thr)
        elif rule.startswith("<"):
            thr = float(rule[1:]); mask_B = (s < thr)
        elif rule.startswith("=="):
            thr = float(rule[2:]); mask_B = (s == thr)
        else:
            raise ValueError("Unknown b_rule")

    # 共同分子
    num = (mask_A & mask_B).sum()

    # 選擇分母（誰是條件）
    if condition.upper() == "B|A":
        den = mask_A.sum()    # P(B|A)
    elif condition.upper() == "A|B":
        den = mask_B.sum()    # P(A|B)
    else:
        raise ValueError("condition must be 'B|A' or 'A|B'")

    if den < min_den:
        return float('nan')

    # 對稱的拉普拉斯平滑
    return float((num + alpha) / (den + 2 * alpha))
