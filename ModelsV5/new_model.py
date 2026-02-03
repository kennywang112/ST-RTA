import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures
from imblearn.under_sampling import RandomUnderSampler


def get_interaction(X, for_poly=[], interaction_type='multiply'):
    if not for_poly:
        return X
    
    X_poly = X.copy()
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(X[for_poly])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(for_poly), index=X.index)

    poly_df = poly_df.drop(columns=for_poly, errors='ignore')
    
    return pd.concat([X_poly, poly_df], axis=1)

def get_interaction_2way(X, base_road, base_vehicle, base_person, interaction_type='multiply'):
    return X

def get_interaction_3way(X, base_road, base_vehicle, base_person, interaction_type='multiply'):
    return X

def get_interaction_mixed(X, base_road, base_vehicle, base_person, interaction_type='multiply'):
    return X

def model_preprocess(
        all_features_df, grid_filter=None, for_poly=[], dim='1way',
        base_road=[], base_vehicle=[], base_person=[],
        interaction_type='multiply'
    ):

    if grid_filter is not None:
        new_grid = pd.concat([grid_filter[['COUNTYNAME']], all_features_df], axis=1)
        county_dummies = pd.get_dummies(new_grid['COUNTYNAME'], prefix='county')
        new_grid_encoded = pd.concat([new_grid.drop(['COUNTYNAME'], axis=1), county_dummies], axis=1)
    else:
        new_grid_encoded = all_features_df.copy()

    new_grid_encoded['hotspot'] = new_grid_encoded['gi_category'].apply(
        lambda x: 1 if 'Hot Spot' in str(x) else 0
    )
    y = new_grid_encoded['hotspot'].values
    exclude_cols = ['geometry', 'grid_id', 'accident_count', 'gi_z', 'gi_p', 'gi_category', 'hotspot']
    feature_cols = [c for c in new_grid_encoded.columns if c not in exclude_cols]

    X_raw = new_grid_encoded[feature_cols].copy()

    X_raw = X_raw.astype(float)
    X_raw = np.log1p(X_raw)

    scaler = StandardScaler()
    X_scaled_values = scaler.fit_transform(X_raw)

    X_raw = pd.DataFrame(
        X_scaled_values, 
        columns=feature_cols, 
        index=X_raw.index
    )
    
    if dim == '1way':
        X_interaction = X_raw.copy()
    elif dim == '2way_poly':
        X_interaction = get_interaction(
            X_raw, for_poly, interaction_type=interaction_type)
    elif dim == '2way':
        X_interaction = get_interaction_2way(
            X_raw, base_road=base_road, base_vehicle=base_vehicle, 
            base_person=base_person, interaction_type=interaction_type)
    elif dim == '3way':
        X_interaction = get_interaction_3way(
            X_raw, base_road=base_road, base_vehicle=base_vehicle, 
            base_person=base_person, interaction_type=interaction_type)
    elif dim == 'mixed':
        X_interaction = get_interaction_mixed(
            X_raw, base_road=base_road, base_vehicle=base_vehicle, 
            base_person=base_person, interaction_type=interaction_type)
    else:
        X_interaction = X_raw.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_interaction, y, test_size=0.2, stratify=y, random_state=42
    )

    rus_train = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus_train.fit_resample(X_train, y_train)
    
    print(f"原始訓練集分佈: {np.bincount(y_train)}")
    print(f"降採樣後訓練集: {np.bincount(y_train_resampled)}")

    rus_test = RandomUnderSampler(random_state=42)
    X_test_resampled, y_test_resampled = rus_test.fit_resample(X_test, y_test)

    y_train_resampled = pd.Series(y_train_resampled)

    y_test = pd.Series(y_test, index=X_test.index)
    y_test_resampled = pd.Series(y_test_resampled)

    return X_train_resampled, X_test, y_train_resampled, y_test, X_test_resampled, y_test_resampled, feature_cols