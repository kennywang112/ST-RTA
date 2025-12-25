"""
This model is to process the data for Analyze and Model folder,
including data cleaning, taiwan process, to get `hex_grid` and
`grid_gi`
"""

import os
import pandas as pd

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
analyze_path = os.path.join(parent_dir, "ST-RTA/utils")
os.chdir(analyze_path)

import pandas as pd
import geopandas as gpd
from utils.utils import get_grid, read_data, calculate_gi, read_taiwan_specific

version = 'V1'
ComputedDataVersion = 'V2'

"""
這裡不需要對combined_data篩空間離群的原因是因為他依照taiwan的範圍去做grid的切割，自動然會把離島的部分去掉
"""
combined_data = read_data()
taiwan, _grid_filter = read_taiwan_specific(read_grid=False)

# print('Start Get Grid')
# hex_grid = get_grid(combined_data, hex_size=0.001, threshold=-1)
# taiwan = taiwan.to_crs(hex_grid.crs)  # 確保 CRS 一致
# hex_grid = hex_grid[hex_grid.intersects(taiwan.unary_union)]
# hex_grid.to_csv(f'../ComputedData{ComputedDataVersion}/Grid/hex_grid{version}.csv', index=False)

# print('Start GI')
# grid_gi = calculate_gi(6, hex_grid, adjacency='knn')
# grid_gi.to_csv(f'../ComputedData{ComputedDataVersion}/Grid/grid_gi{version}.csv', index=False)

"""
正常情況第一次不會有grid_filter，所以需要先在這裡執行read_taiwan_specific(read_grid=True)會有的情況
"""
import ast
from shapely import wkt

TM2 = 3826
grid_gi_df = pd.read_csv(f'../ComputedData{ComputedDataVersion}/Grid/grid_gi{version}.csv') 
grid_gi_df['accident_indices'] = grid_gi_df['accident_indices'].apply(ast.literal_eval)
grid_gi_df['geometry'] = grid_gi_df['geometry'].apply(wkt.loads)
grid_gi  = gpd.GeoDataFrame(grid_gi_df, geometry='geometry').set_crs(TM2, allow_override=True)
grid_gi['geometry'] = grid_gi.geometry.centroid

taiwan_cnty = taiwan[['COUNTYNAME','geometry']].dissolve(by='COUNTYNAME')
taiwan_cnty['geometry'] = taiwan_cnty.buffer(0)
county_join = gpd.sjoin(grid_gi[['geometry']], taiwan_cnty, how='left', predicate='within')
grid_gi['COUNTYNAME'] = county_join['COUNTYNAME']
# 這些都是離島資料，因為在taiwan被篩選掉了，所以會因為對應不到所以回傳空值
grid_filter = grid_gi[grid_gi['accident_indices'].str.len() > 0]
grid_filter.reset_index(inplace=True)


"""
This is for model feature concat
logic: get all select group, which is then convert to proportion
"""
print('Start Extract')

from utils.config import select_group
from utils.utils_model import extract_features

all_features_list = []

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

all_features_df.to_csv(f"../ComputedData{ComputedDataVersion}/ForModel/all_features{version}.csv", index=False)
