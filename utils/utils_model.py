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