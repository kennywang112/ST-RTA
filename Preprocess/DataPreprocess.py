"""
This model is to process the data for Analyze and Model folder,
including data cleaning, taiwan process, to get `hex_grid` and
`grid_gi`
"""

import pandas as pd
import geopandas as gpd
from utils.utils import get_grid, read_data, calculate_gi

version = 'V2'

combined_data = read_data()
taiwan = gpd.read_file('./Data/OFiles_9e222fea-bafb-4436-9b17-10921abc6ef2/TOWN_MOI_1140318.shp')
taiwan = taiwan[(~taiwan['TOWNNAME'].isin(['旗津區', '頭城鎮', '蘭嶼鄉', '綠島鄉', '琉球鄉'])) & 
                (~taiwan['COUNTYNAME'].isin(['金門縣', '連江縣', '澎湖縣']))]

print('Start Get Grid')
hex_grid = get_grid(combined_data, hex_size=0.001, threshold=-1)
taiwan = taiwan.to_crs(hex_grid.crs)  # 確保 CRS 一致
hex_grid = hex_grid[hex_grid.intersects(taiwan.unary_union)]
hex_grid.to_csv(f'./ComputedData/Grid/hex_grid{version}.csv', index=False)

print('Start GI')
grid_gi = calculate_gi(6, hex_grid, adjacency='knn')
grid_gi.to_csv(f'./ComputedData/Grid/grid_gi{version}.csv', index=False)

"""
This is for model feature concat
logic: get all select group, which is then convert to proportion
"""
print('Start Extract')

from config import select_group
from utils_model import extract_features

all_features_list = []

grid_filter = grid_gi[grid_gi['accident_indices'].str.len() > 0]
grid_filter.reset_index(inplace=True)

for rows in range(grid_filter.shape[0]):
    features = extract_features(grid_filter, combined_data, select_group, rows)
    all_features_list.append(features)

all_features_df = pd.concat(all_features_list, ignore_index=True)
all_features_df.fillna(0, inplace=True)

all_features_df['original_speed'] = all_features_df['速限-第1當事者_mean']
all_features_df[['mrt_100m_count_mean', 'youbike_100m_count_mean', 'parkinglot_100m_count_mean', '速限-第1當事者_mean']] =\
      all_features_df[['mrt_100m_count_mean', 'youbike_100m_count_mean', 'parkinglot_100m_count_mean', '速限-第1當事者_mean']].\
        apply(lambda x: (x - x.min()) / (x.max() - x.min()))

all_features_df['hotspot'] = grid_filter['hotspot']

all_features_df.to_csv(f"./ComputedData/ForModel/all_features{version}.csv", index=False)
