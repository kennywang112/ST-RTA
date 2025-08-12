import json
import folium
import numpy as np
from esda import Moran, G_Local
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Polygon
from libpysal.weights import DistanceBand, Queen, KNN

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
    hex_size = 1  # 度數，1 度 ≈ 111 公里
    """
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['經度'], data['緯度']))
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    gdf = gdf[
        (gdf['經度'] >= 119.7) & (gdf['經度'] <= 122.1) &
        (gdf['緯度'] >= 21.8) & (gdf['緯度'] <= 25.4)
    ]
    # 計算範圍 (bounding box)
    if specific_area is not None:
        bounds = specific_area.to_crs(epsg=4326).total_bounds  # (minx, miny, maxx, maxy)
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

    # 將事故點分配到 hexagon
    joined = gpd.sjoin(gdf, hex_grid, how='left', predicate='within')

    def calculate_weighted_accidents(group):
        # 計算加權事故數量
        return group.apply(lambda row: row['num_accidents'] * 2.714 if row['source'] == 'A1' else row['num_accidents'], axis=1).sum()

    hex_grid['num_accidents'] = joined.groupby('index_right').apply(calculate_weighted_accidents)

    # 每個 hexagon 內事故的原始索引 list
    hex_grid['accident_indices'] = hex_grid.index.map(lambda idx: list(joined.index[joined['index_right'] == idx]))

    # 沒有事故的 hexagon 設為 0
    hex_grid['num_accidents'] = hex_grid['num_accidents'].fillna(0)#.astype(int)
    hex_grid['accident_indices'] = hex_grid['accident_indices'].apply(lambda x: x if isinstance(x, list) else [])

    hex_grid = hex_grid[hex_grid['num_accidents'] > threshold]
    hex_grid.to_crs(epsg=3826, inplace=True)

    return hex_grid

def specific_polygon(df, taiwan, county=None):

    if county:
        taiwan_specific = taiwan[taiwan['COUNTYNAME'].isin(county)]
    else:
        taiwan_specific = taiwan

    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['經度'], df['緯度']),
        crs="EPSG:4326"  # WGS84
    )

    # 轉到和 taiwan 一樣的座標系
    gdf_points = gdf_points.to_crs(taiwan_specific.crs)

    # 做空間篩選 — 找在 taiwan Polygon 裡的點
    # 用 spatial join
    points_in_taiwan = gpd.sjoin(gdf_points, taiwan_specific, predicate='within', how='inner')

    # points_in_taiwan 現在就是落在台北/新北/桃園的點
    # 如果要拿回原本的欄位：
    specific_A2 = points_in_taiwan.drop(columns=['index_right'])  # sjoin多的欄位可以丟掉
    print(specific_A2.shape)
    taiwan_specific.to_crs(epsg=3826, inplace=True)  # 轉到公尺座標系

    return specific_A2, taiwan_specific

def calculate_gi(best_distance, grid, adjacency=None):

    # 建立鄰接矩陣（以中心點）
    centroids = grid.centroid
    coords = np.array(list(zip(centroids.x, centroids.y)))

    if adjacency=='knn':
        w = KNN.from_array(coords, k=best_distance)
    elif adjacency=='queen':
        w = Queen.from_dataframe(grid)
    else:
        w = DistanceBand(coords, threshold=best_distance, binary=True, silence_warnings=True)

    w.transform = 'r'
    y = grid['num_accidents'].values
    g_local = G_Local(y, w)
    grid['GiZScore'] = g_local.Zs

    grid['hotspot'] = 'Not Significant'
    grid.loc[grid['GiZScore'] > 2.58, 'hotspot'] = 'Hotspot 99%'
    grid.loc[(grid['GiZScore'] > 1.96) & (grid['GiZScore'] <= 2.58), 'hotspot'] = 'Hotspot 95%'
    grid.loc[(grid['GiZScore'] > 1.65) & (grid['GiZScore'] <= 1.96), 'hotspot'] = 'Hotspot 90%'
    grid.loc[grid['GiZScore'] < -2.58, 'hotspot'] = 'Coldspot 99%'
    grid.loc[(grid['GiZScore'] >= -2.58) & (grid['GiZScore'] < -1.96), 'hotspot'] = 'Coldspot 95%'
    grid.loc[(grid['GiZScore'] >= -1.96) & (grid['GiZScore'] < -1.65), 'hotspot'] = 'Coldspot 90%'

    return grid

def plot_hex_grid(specific, taiwan_specific, threshold=0, hex_size=0.01):

    hex_grid = get_grid(specific, taiwan_specific, hex_size, threshold)
    hex_grid = hex_grid[hex_grid.intersects(taiwan_specific.unary_union)]

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

def incremental_spatial_autocorrelation(gdf, value_col, min_dist=1000, max_dist=50000, step=1000):
    """
    gdf: CRS (meter)。
    value_col: 你要做 spatial autocorrelation 的欄位，如num_accidents
    min_dist: 最小距離 (公尺)
    max_dist: 最大距離 (公尺)
    step: 間隔 (公尺)
    """

    thresholds = np.arange(min_dist, max_dist + step, step)

    moran_I = []
    z_scores = []
    p_values = []

    centroids = gdf.centroid
    coords = np.array(list(zip(centroids.x, centroids.y)))
    y = gdf[value_col].values

    for thresh in thresholds:
        print(f"Processing threshold: {thresh} meters")
        w = DistanceBand(coords, threshold=thresh, binary=True, silence_warnings=True)
        w.transform = 'r'  # row-standardized
        moran = Moran(y, w)
        print(moran.z_norm)

        moran_I.append(moran.I)
        z_scores.append(moran.z_norm)  # ArcGIS 用的是 Z-score
        p_values.append(moran.p_norm)

    return thresholds, moran_I, z_scores, p_values

def incremental_spatial_autocorrelation_knn(gdf, value_col, min_k=2, max_k=50, step=1):
    """
    gdf: CRS (meter)。
    value_col: 你要做 spatial autocorrelation 的欄位，如 num_accidents。
    min_k: 最小鄰居數。
    max_k: 最大鄰居數。
    step: 每次增加幾個鄰居。
    """

    ks = np.arange(min_k, max_k + step, step)

    moran_I = []
    z_scores = []
    p_values = []

    centroids = gdf.centroid
    coords = np.array(list(zip(centroids.x, centroids.y)))
    y = gdf[value_col].values

    for k in ks:
        print(f"Processing k = {k} neighbors")
        w = KNN.from_array(coords, k=k)
        w.transform = 'r'  # row-standardized
        moran = Moran(y, w)
        print(moran.z_norm)

        moran_I.append(moran.I)
        z_scores.append(moran.z_norm)  # ArcGIS 用的是 Z-score
        p_values.append(moran.p_norm)

    return ks, moran_I, z_scores, p_values

def get_isa_plot(df, threshold=0, hex_size=0.01):

    hex_grid = get_grid(df, hex_size)
    hex_grid = hex_grid[hex_grid['num_accidents'] > threshold] 

    # 投影成公尺
    hex_grid = hex_grid.to_crs(epsg=3826)

    # Incremental Spatial Autocorrelation
    # grid 中心點到中心點的incremental
    thresholds, moran_I, z_scores, p_values = incremental_spatial_autocorrelation(
        hex_grid, value_col='num_accidents', min_dist=1000, max_dist=10000, step=1000
    )

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds / 1000, z_scores, marker='o')
    plt.xlabel('Distance Threshold (km)')
    plt.ylabel('Z-Score')
    plt.title("Incremental Spatial Autocorrelation (ISA)")
    plt.grid(True)
    plt.show()

    z_scores_array = np.array(z_scores)
    thresholds_array = np.array(thresholds)
    moran_I_array = np.array(moran_I)

    # 找不是 nan 的位置
    valid_mask = ~np.isnan(z_scores_array)

    # 篩選掉 nan
    z_scores_valid = z_scores_array[valid_mask]
    thresholds_valid = thresholds_array[valid_mask]
    moran_I_valid = moran_I_array[valid_mask]

    # 找最大 Z-score 的 index
    best_idx = np.argmax(z_scores_valid)
    best_distance = thresholds_valid[best_idx]
    print(f"最佳分析距離 (m): {best_distance}")
    print(f"Z-score: {z_scores_valid[best_idx]:.4f}")
    print(f"Moran's I: {moran_I_valid[best_idx]:.4f}")

    return best_distance

def plot_gi(taiwan, grid):

    cmap = mcolors.ListedColormap([
        '#800026',  # dark red - Hotspot 99%
        '#FC4E2A',  # red - Hotspot 95%
        '#FD8D3C',  # light red - Hotspot 90%
        '#d9d9d9',  # grey - Not Significant
        '#6baed6',  # light blue - Coldspot 90%
        '#3182bd',  # blue - Coldspot 95%
        '#08519c'   # dark blue - Coldspot 99%
    ])

    # 會照順序排
    categories = [
        'Hotspot 99%', 
        'Hotspot 95%', 
        'Hotspot 90%', 
        'Not Significant', 
        'Coldspot 90%', 
        'Coldspot 95%', 
        'Coldspot 99%'
    ]

    fig, ax = plt.subplots(figsize=(10, 10))
    taiwan.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5)

    grid.plot(
        column='hotspot', 
        categorical=True, 
        cmap=cmap, 
        legend=True, 
        edgecolor='grey', 
        linewidth=0.2, 
        alpha=0.6,
        ax=ax,
        categories=categories,
        legend_kwds={
            'bbox_to_anchor': (1.05, 1),  # legend移出主圖右邊
            'loc': 'upper left',          # 從左上角開始排列
            'frameon': False              # 不要框線，看起來乾淨
        }
    )

    plt.title('Hotspot Analysis (Getis-Ord Gi*) - 90%, 95%, 99% Confidence Levels')
    plt.axis('off')
    plt.show()

def plot_map(data, grid, gi=False, count=None):
    # grid = grid[['hotspot', 'num_accidents', 'mrt_count', 'geometry']].copy()
    grid = grid.copy()
    grid = grid.drop(columns=['centroid'], errors='ignore') # 確保不會有centroid欄位，無法被序列化
    grid_json = json.loads(grid.to_json())

    # 地圖中心點
    center = [data['緯度'].mean(), data['經度'].mean()]

    # 英文版
    m = folium.Map(
        location=center, 
        zoom_start=10, 
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri'
    )

    # 建立底圖
    # m = folium.Map(location=center, zoom_start=10, tiles='OpenStreetMap')

    if gi:
        # 加入格網
        folium.GeoJson(
            grid_json,
            style_function=lambda feature: {
                'fillColor': '#ff0000' if feature['properties']['hotspot'] == 'Hotspot 99%' else
                            '#ff6666' if feature['properties']['hotspot'] == 'Hotspot 95%' else
                            '#ffe6e6' if feature['properties']['hotspot'] == 'Hotspot 90%' else
                            '#cccccc' if feature['properties']['hotspot'] == 'Not Significant' else
                            '#6666ff' if feature['properties']['hotspot'] == 'Coldspot 90%' else
                            '#3399ff' if feature['properties']['hotspot'] == 'Coldspot 95%' else
                            '#0000ff',
                'color': 'grey',
                'weight': 0.5,
                'fillOpacity': 0.6
            }
        ).add_to(m)
    else:
        folium.GeoJson(
            grid_json,
            style_function=lambda feature: {
                'fillColor': '#FFEDA0' if feature['properties']['num_accidents'] < 100 else
                            '#FEB24C' if feature['properties']['num_accidents'] < 500 else
                            '#F03B20',
                'color': 'grey',
                'weight': 0.5,
                'fillOpacity': 0.6
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['num_accidents'],
                aliases=['Accidents:'],
                localize=True
            )
        ).add_to(m)

    if count is not None:
        for _, row in grid.iterrows():
            if row[count] > 0:  # 只顯示有站點的多邊形
                centroid = row['geometry'].centroid  # 獲取多邊形的中心點
                folium.CircleMarker(
                    location=[centroid.y, centroid.x],  # 中心點的經緯度
                    radius=row[count] * 1,  # 半徑根據特徵調整，放大 2 倍
                    color='#9a9af5',  # 邊框顏色
                    fill=True,
                    fill_color='#9a9af5',  # 填充顏色
                    fill_opacity=0.5,
                    tooltip=f"Count: {row[count]}"  # 提示顯示站點數量
                ).add_to(m)

    return m