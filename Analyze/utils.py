
import numpy as np
from esda import Moran, G_Local
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Polygon
from libpysal.weights import DistanceBand

def create_hexagon(center_x, center_y, size):
    angles = np.linspace(0, 2 * np.pi, 7)
    return Polygon([
        (center_x + size * np.cos(angle), center_y + size * np.sin(angle))
        for angle in angles
    ])

def get_grid(data, hex_size, threshold=0):
    """
    # hexagon 大小 (degree)
    # 台灣約395,144 km
    hex_size = 1  # 度數，1 度 ≈ 111 公里
    """
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['經度'], data['緯度']))

    gdf = gdf[
        (gdf['經度'] >= 119.7) & (gdf['經度'] <= 122.1) &
        (gdf['緯度'] >= 21.8) & (gdf['緯度'] <= 25.4)
    ]
    # 計算範圍 (bounding box)
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
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
    # 將事故點分配到 hexagon
    joined = gpd.sjoin(gdf, hex_grid, how='left', predicate='within')
    # 統計每個 hexagon 的事故數量
    hex_grid['num_accidents'] = joined.groupby('index_right').size()
    # 沒有事故的 hexagon 設為 0
    hex_grid['num_accidents'] = hex_grid['num_accidents'].fillna(0).astype(int)
    hex_grid = hex_grid[hex_grid['num_accidents'] > threshold]

    return hex_grid

def specific_polygon(df, taiwan, county):

    taiwan_specific = taiwan[taiwan['COUNTYNAME'].isin(county)]

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

    return specific_A2, taiwan_specific

def calculate_gi(best_distance, grid):

    # 建立鄰接矩陣（以中心點）
    centroids = grid.centroid
    coords = np.array(list(zip(centroids.x, centroids.y)))

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

    print(grid[['num_accidents', 'GiZScore']].head())

    return grid

def plot_hex_grid(specific_A2, taiwan_specific, threshold=0, hex_size=0.01):

    hex_grid = get_grid(specific_A2, hex_size)

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
    plt.title('Hexagon Accident Counts (num_accidents > 0)')
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