import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import box

TM2 = 3826
computeddata = 'ComputedDataV2'

def read_data():
    dataA1 = pd.read_csv(f'../{computeddata}/Accident/DataA1_with_MYP.csv')
    dataA2 = pd.read_csv(f'../{computeddata}/Accident/DataA2_with_MYP.csv')

    filtered_A2 = dataA2[dataA2['當事者順位'] == 1].copy()
    filtered_A1 = dataA1[dataA1['當事者順位'] == 1].copy()

    filtered_A1['source'] = 'A1'
    filtered_A2['source'] = 'A2'
    filtered_A1['num_accidents'] = 1 
    filtered_A2['num_accidents'] = 1
    combined_data = pd.concat([filtered_A1, filtered_A2], ignore_index=True)

    # 替換離群值成中位數
    median_speed = combined_data.loc[combined_data['速限-第1當事者'] < 200, '速限-第1當事者'].median()
    median_age = combined_data.loc[(combined_data['當事者事故發生時年齡'] > 0) & 
                                   (combined_data['當事者事故發生時年齡'] < 100),
                    '當事者事故發生時年齡'].median()
    combined_data.loc[combined_data['速限-第1當事者'] >= 200, '速限-第1當事者'] = median_speed
    combined_data.loc[(combined_data['當事者事故發生時年齡'] >= 100) | 
                    (combined_data['當事者事故發生時年齡'] <= 0), '當事者事故發生時年齡'] = median_age

    return combined_data

import ast
import geopandas as gpd
from shapely import wkt

def read_taiwan_specific(read_grid=False):
    taiwan = gpd.read_file('../Data/OFiles_9e222fea-bafb-4436-9b17-10921abc6ef2/TOWN_MOI_1140318.shp')
    taiwan = taiwan[(~taiwan['TOWNNAME'].isin(['蘭嶼鄉', '綠島鄉', '琉球鄉'])) & 
                    (~taiwan['COUNTYNAME'].isin(['金門縣', '連江縣', '澎湖縣']))].to_crs(TM2)

    minx, miny, maxx, maxy = taiwan.total_bounds
    clip_box = box(minx, 2400000, 380000, maxy)
    clipper = gpd.GeoDataFrame(geometry=[clip_box], crs=taiwan.crs)
    taiwan = gpd.clip(taiwan, clipper)

    if read_grid:
        taiwan_cnty = taiwan[['COUNTYNAME','geometry']].dissolve(by='COUNTYNAME')
        taiwan_cnty['geometry'] = taiwan_cnty.buffer(0)

        # 原始以 0.001 grid 計算出的區域事故及對應索引, 依照 hex_grid 計算出來的GI
        grid_gi_df = pd.read_csv(f'../{computeddata}/Grid/grid_giV1.csv')
        grid_gi_df['accident_indices'] = grid_gi_df['accident_indices'].apply(ast.literal_eval)
        grid_gi_df['geometry'] = grid_gi_df['geometry'].apply(wkt.loads)
        grid_gi  = gpd.GeoDataFrame(grid_gi_df, geometry='geometry').set_crs(TM2, allow_override=True)
        grid_gi['geometry'] = grid_gi.geometry.centroid

        county_join = gpd.sjoin(grid_gi[['geometry']], taiwan_cnty, how='left', predicate='within')
        grid_gi['COUNTYNAME'] = county_join['COUNTYNAME']
        # 這些都是離島資料，因為在taiwan被篩選掉了，所以會因為對應不到所以回傳空值
        grid_filter = grid_gi[grid_gi['accident_indices'].str.len() > 0]
        grid_filter.reset_index(inplace=True)

    else:
        grid_filter = None

    return taiwan, grid_filter

def create_hexagon(center_x, center_y, size):
    angles = np.linspace(0, 2 * np.pi, 7)
    return Polygon([
        (center_x + size * np.cos(angle), center_y + size * np.sin(angle))
        for angle in angles
    ])

def get_grid(data, specific_area=None, hex_size=0.01, threshold=0):
    """
    # hexagon 大小 (degree)
    # 台灣約395,144 km
    hex_size = 1 度
    """

    if isinstance(data, gpd.GeoDataFrame) and 'geometry' in data.columns:
        gdf = data.copy()
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        gdf = gdf.to_crs(epsg=4326)
    else:
        gdf = gpd.GeoDataFrame(data.copy(), geometry=gpd.points_from_xy(data['經度'], data['緯度']), crs='EPSG:4326')

    gdf = gdf[
        (gdf['經度'] >= 119.7) & (gdf['經度'] <= 122.1) &
        (gdf['緯度'] >= 21.8) & (gdf['緯度'] <= 25.4)
    ]
    # 計算範圍 (bounding box)
    if specific_area is not None:
        bounds = specific_area.to_crs(epsg=4326).total_bounds # (minx, miny, maxx, maxy)
    else:
        bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds

    # 六邊形的寬度和高度
    width = hex_size * 2
    height = np.sqrt(3) * hex_size

    # 計算橫向和縱向的hexagon間距
    x_spacing = width * 3/4
    y_spacing = height

    hexagons = []

    print('create hexagon')
    x = minx
    while x < maxx + width:
        y = miny
        while y < maxy + height:
            hex_center = (x, y)
            hexagon = create_hexagon(*hex_center, hex_size)
            hexagons.append(hexagon)
            y += y_spacing
        x += x_spacing

    # 每列交錯排列，蜂巢狀
    hexagons_shifted = []
    row = 0
    x = minx
    while x < maxx + width:
        y = miny + (y_spacing / 2 if row % 2 else 0) 
        while y < maxy + height:
            hex_center = (x, y)
            hexagon = create_hexagon(*hex_center, hex_size)
            hexagons_shifted.append(hexagon)
            y += y_spacing
        x += x_spacing
        row += 1

    print('get grid')
    hex_grid = gpd.GeoDataFrame(geometry=hexagons_shifted, crs='EPSG:4326')
    hex_grid = hex_grid.to_crs(gdf.crs)

    gdf = gdf.drop(columns=['index_right', 'index_left'], errors='ignore')
    hex_grid = hex_grid.drop(columns=['index_right', 'index_left'], errors='ignore')
    # 將事故點分配到 hexagon
    joined = gpd.sjoin(gdf, hex_grid, how='left', predicate='within')

    weights = np.where(joined['source'] == 'A1', 2.714, 1.0)
    joined['weighted'] = joined['num_accidents'] * weights
    num_by_hex = joined.groupby('index_right')['weighted'].sum()
    hex_grid['num_accidents'] = num_by_hex.reindex(hex_grid.index, fill_value=0)

    # 每個 hexagon 內事故的原始索引 list
    idx_map = joined.groupby('index_right').apply(lambda s: list(s.index))
    hex_grid['accident_indices'] = idx_map.reindex(hex_grid.index).apply(lambda x: x if isinstance(x, list) else [])

    hex_grid = hex_grid[hex_grid['num_accidents'] > threshold].copy()
    hex_grid = hex_grid.to_crs(epsg=TM2)
    
    return hex_grid

def specific_polygon(df, taiwan, county=None):

    if county:
        taiwan_specific = taiwan[taiwan['COUNTYNAME'].isin(county)]
    else:
        taiwan_specific = taiwan

    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['經度'], df['緯度']),
        crs="EPSG:4326" # WGS84
    )
    gdf_points = gdf_points.to_crs(taiwan_specific.crs)
    points_in_taiwan = gpd.sjoin(gdf_points, taiwan_specific, predicate='within', how='inner')

    specific = points_in_taiwan.drop(columns=['index_right']) # sjoin多的欄位可以丟掉
    taiwan_specific.to_crs(epsg=TM2, inplace=True) # 轉到公尺座標系

    return specific, taiwan_specific

def plot_hex_grid(specific, taiwan_specific, threshold=0, hex_size=0.01):

    hex_grid = get_grid(specific, taiwan_specific, hex_size, threshold)
    taiwan_specific_tm2 = taiwan_specific.to_crs(epsg=TM2)
    hex_grid = hex_grid[hex_grid.intersects(taiwan_specific_tm2.unary_union)]

    fig, ax = plt.subplots(figsize=(10, 10))
    taiwan_specific.plot(ax=ax, color='white', edgecolor='black')  # 底圖：台灣行政區
    hex_grid.plot(
        column='num_accidents', 
        cmap='OrRd', 
        legend=True, 
        edgecolor='black', 
        linewidth=0.2, 
        alpha=0.6,
        ax=ax
    )
    plt.title('Hexagon Accident Counts')
    plt.axis('off')
    plt.show()

    return hex_grid
