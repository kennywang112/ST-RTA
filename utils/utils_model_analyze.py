import matplotlib.pyplot as plt
import numpy as np

def human_vs_road(importance):

    out = {}

    for name, v in importance.items():
        if 'x' not in name or 'cause-group' not in name:
            continue

        base, cause_tag = name.split(" x ", 1)
        out.setdefault(cause_tag, {})[base] = v[1]

    return out

def car_vs_road(importance):

    out = {}

    for name, v in importance.items():
        if 'x' not in name or '車種' not in name:
            continue
        
        base, cause_tag = name.split(" x ", 1)
        out.setdefault(cause_tag, {})[base] = v[1]

    return out

def human_vs_car(importance):

    out = {}

    for name, v in importance.items():
        if 'x' not in name or 'cause-group' not in name or '車種' not in name:
            continue

        base, cause_tag = name.split(" x ", 1)

        out.setdefault(cause_tag, {})[base] = v[1]

    return out

def plot_interaction(contain_str, structured_cause, countycity_dct=None, col_translation=None, filter_val=None, top_k=100):
    odds_df_cause = structured_cause[structured_cause.columns[structured_cause.columns.str.contains(contain_str)]]
    odds_df_cause.sort_values(by=odds_df_cause.columns[0], ascending=False)

    if countycity_dct and col_translation:
        odds_df_cause = odds_df_cause.rename(columns=countycity_dct, index=col_translation)
    if filter_val:
        odds_df_cause = odds_df_cause[odds_df_cause[odds_df_cause.columns[0]] < filter_val]

    plt.rcParams['font.family'] = ['Arial Unicode Ms']
    df_long_cause = odds_df_cause.stack().reset_index()
    df_long_cause.columns = ['feature', 'cause_group', 'value']
    rank_cause = (
        df_long_cause.assign(abs_log=lambda d: np.abs(np.log(d['value'])))
            .groupby('feature')['abs_log'].max()
            .sort_values(ascending=False)
    )
    feat_sel_cause = rank_cause.head(top_k).index
    plot_data_cause = df_long_cause[df_long_cause['feature'].isin(feat_sel_cause)]
    features_cause = plot_data_cause['feature'].dropna().unique().tolist()
    features_cause = [f for f in rank_cause.index if f in features_cause]
    ypos_cause = {f: i for i, f in enumerate(features_cause)}  
    plt.figure(figsize=(14, max(6, 0.4*len(features_cause))))
    plt.axvline(1.0, linestyle='--', linewidth=1)
    plt.axvline(1.5, linestyle='--', linewidth=1)
    plt.axvline(0.5, linestyle='--', linewidth=1)
    plt.axvline(2, linestyle='--', linewidth=1)
    for cause_group, d in plot_data_cause.groupby('cause_group'):
        y = [ypos_cause[f] for f in d['feature']]
        plt.scatter(d['value'], y, s=20, label=cause_group)
    plt.yticks(range(len(features_cause)), features_cause)
    plt.xlabel('Odds Ratio (OR)')
    plt.ylabel('Feature')
    plt.title('Effect per feature (x = OR)')
    plt.legend(title=contain_str, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()