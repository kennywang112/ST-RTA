"""
This model is to get the county/city name for each record in the combined data.
"""

import os
import pandas as pd

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
analyze_path = os.path.join(parent_dir, "ST-RTA/utils")
os.chdir(analyze_path)

import pandas as pd
import geopandas as gpd
from utils.utils import read_data, read_taiwan_specific
from shapely.geometry import Point

V = '2'

combined_data = read_data()
taiwan, grid_filter = read_taiwan_specific(read_grid=False)
taiwan_cnty = taiwan[['COUNTYNAME','geometry']].dissolve(by='COUNTYNAME')
taiwan_cnty['geometry'] = taiwan_cnty.buffer(0)
cnty = taiwan_cnty.reset_index()

gdf_points = gpd.GeoDataFrame(combined_data, 
                              geometry=[Point(xy) for xy in zip(combined_data['經度'], combined_data['緯度'])],
                              crs='EPSG:4326')

gdf_joined = gpd.sjoin(gdf_points, cnty[['COUNTYNAME', 'geometry']], how='left', predicate='within')
combined_data['COUNTYNAME'] = gdf_joined['COUNTYNAME'].values

combined_data.to_csv(f'../ComputedDataV{V}/Accident/combineddata_with_CC.csv', index=False)