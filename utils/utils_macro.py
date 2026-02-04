import json
import folium
import libpysal
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from esda.getisord import G_Local
from esda.moran import Moran_Local, Moran
from libpysal.weights import Queen, DistanceBand, KNN
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from matplotlib.font_manager import FontProperties

class LocalMoranAnalysis:
    def __init__(self, hex_grid, taiwan, k=6):
        """
        初始化 Local Moran Analysis 類別
        :param hex_grid: GeoDataFrame，包含六邊形網格和事故數量
        :param taiwan: GeoDataFrame，台灣地圖資料
        :param k: int，KNN 的鄰居數量
        """
        self.hex_grid = hex_grid
        self.taiwan = taiwan
        self.k = k
        self.moran_local = None
        self.y = None
        self.w_knn = None

    def calculate_local_moran(self):
        """
        計算 Local Moran's I 和相關統計
        """
        self.hex_grid['centroid'] = self.hex_grid.geometry.centroid
        coords = np.vstack((self.hex_grid['centroid'].x, self.hex_grid['centroid'].y)).T

        # KNN 權重
        self.w_knn = KNN.from_array(coords, k=self.k)
        self.w_knn.transform = 'r'  # row-standardized

        self.y = self.hex_grid['num_accidents'].values
        self.moran_local = Moran_Local(self.y, self.w_knn)

        self.hex_grid['local_I'] = self.moran_local.Is
        self.hex_grid['local_p'] = self.moran_local.p_sim
        self.hex_grid['significant'] = (self.hex_grid['local_p'] < 0.05)

    def plot_lisa(self):
        """
        繪製 LISA 群組圖
        """
        # 群組：高-高, 低-低, 高-低, 低-高
        quadrant = []
        for i in range(len(self.y)):
            if self.moran_local.q[i] == 1:
                quadrant.append('High-High')
            elif self.moran_local.q[i] == 3:
                quadrant.append('Low-Low')
            elif self.moran_local.q[i] == 2:
                quadrant.append('High-Low')
            elif self.moran_local.q[i] == 4:
                quadrant.append('Low-High')
            else:
                quadrant.append('Not Significant')

        self.hex_grid['quadrant'] = quadrant

        # 畫圖
        fig, ax = plt.subplots(figsize=(6, 10))

        self.taiwan.plot(ax=ax, color='white', edgecolor='black') 
        colors = {
            'High-High': '#8c1004',
            'Low-Low': '#04468c',
            'High-Low': '#f06e62',
            'Low-High': '#65a8f0',
            'Not Significant': '#c9c2c1'
        }

        for quad, color in colors.items():
            subset = self.hex_grid[self.hex_grid['quadrant'] == quad]
            ax.scatter(
                subset['centroid'].x, 
                subset['centroid'].y, 
                color=color, 
                label=quad, 
                s=10,
                alpha=0.3
            )

        ax.legend()
        ax.set_title('Local Moran\'s I (KNN-based LISA)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.show()

    def lisa_scatter_plot(self):
        """
        繪製 Moran Scatter Plot
        """
        # 原始值
        y = self.hex_grid['num_accidents'].values.reshape(-1, 1)

        # 標準化
        scaler = StandardScaler()
        z = scaler.fit_transform(y).flatten()        # 自己的 Z-score
        wz = self.w_knn.sparse.dot(z)                # 鄰居的 Z-score 加權平均

        fig, ax = plt.subplots(figsize=(8, 8))

        plt.axhline(0, color='k', linestyle='--')
        plt.axvline(0, color='k', linestyle='--')

        colors = []
        for zi, wzi in zip(z, wz):
            if zi > 0 and wzi > 0:
                colors.append('#8c1004')          # High-High
            elif zi < 0 and wzi < 0:
                colors.append('#04468c')         # Low-Low
            elif zi > 0 and wzi < 0:
                colors.append('#f06e62')         # High-Low
            elif zi < 0 and wzi > 0:
                colors.append('#65a8f0')         # Low-High
            else:
                colors.append('#c9c2c1')

        ax.scatter(z, wz, c=colors, s=10, alpha=0.7)

        # 回歸線 (slope = global Moran's I)
        b, a = np.polyfit(z, wz, 1)
        xfit = np.linspace(min(z), max(z), 100)
        yfit = a + b * xfit
        ax.plot(xfit, yfit, color='#a0bcd9', linewidth=2, label=f'Regression Line\nSlope={b:.3f}')

        plt.xlabel('Standardized Num of Accidents (Z-score)')
        plt.ylabel('Spatial Lag of Accidents (Z-score)')
        plt.title('Moran Scatter Plot (KNN LISA, Standardized)')
        plt.legend()
        plt.show()

    def plot_lisa_folium(self):

        color_map = {
            'High-High': '#8c1004',
            'Low-Low': '#04468c',
            'High-Low': '#f06e62',
            'Low-High': '#65a8f0',
            'Not Significant': '#c9c2c1'
        }
        grid_wgs = self.hex_grid.to_crs(epsg=4326)
        grid_wgs = grid_wgs[['quadrant', 'num_accidents', 'geometry']].copy()
        grid_json = json.loads(grid_wgs.to_json())
        center = [grid_wgs.geometry.centroid.y.mean(), grid_wgs.geometry.centroid.x.mean()]
        m = folium.Map(location=center, zoom_start=10, tiles='CartoDB Voyager')
        folium.GeoJson(
            grid_json,
            style_function=lambda feature: {
                'fillColor': color_map.get(feature['properties']['quadrant'], '#c9c2c1'),
                'color': 'grey',
                'weight': 0.1,
                'fillOpacity': 0.5
            },
            tooltip=folium.GeoJsonTooltip(fields=['quadrant', 'num_accidents'])
        ).add_to(m)
        return m

class GetisOrdGiAnalysis:
    def __init__(self, grid, taiwan):
        self.grid = grid
        self.taiwan = taiwan

    def calculate_gi(self, best_distance, adjacency=None):
        """
        best_distance: when adjacency is 'knn', it indicates the number of neighbors k, 
                        if adjacency is 'distance', it indicates the distance threshold
        grid: GeoDataFrame with 'num_accidents' column
        adjacency: 'knn', 'queen', or 'distance'
        Returns: GeoDataFrame with 'GiZScore' column added
        """
        centroids = self.grid.centroid
        coords = np.array(list(zip(centroids.x, centroids.y)))

        if adjacency=='knn':
            w = KNN.from_array(coords, k=best_distance)
        elif adjacency=='queen':
            w = Queen.from_dataframe(self.grid)
        else:
            w = DistanceBand(coords, threshold=best_distance, binary=True, silence_warnings=True)

        if w.islands:
            self.grid = self.grid.drop(index=w.islands)
            return self.calculate_gi(best_distance, adjacency) 

        w = libpysal.weights.util.fill_diagonal(w, 1)
        w.transform = 'r'
        y = self.grid['num_accidents'].astype(np.float64).values
        g_local = G_Local(y, w, star=True)
        self.grid['GiZScore'] = g_local.Zs
        self.grid['GiPValue'] = g_local.p_sim

        self.grid['hotspot'] = 'Not Significant'
        self.grid.loc[(self.grid['GiPValue'] < 0.01) & (self.grid['GiZScore'] > 0), 'hotspot'] = 'Hotspot 99%'
        self.grid.loc[(self.grid['GiPValue'] < 0.05) & (self.grid['GiZScore'] > 0) & (self.grid['hotspot'] == 'Not Significant'), 'hotspot'] = 'Hotspot 95%'
        self.grid.loc[(self.grid['GiPValue'] < 0.10) & (self.grid['GiZScore'] > 0) & (self.grid['hotspot'] == 'Not Significant'), 'hotspot'] = 'Hotspot 90%'
        self.grid.loc[(self.grid['GiPValue'] < 0.01) & (self.grid['GiZScore'] < 0), 'hotspot'] = 'Coldspot 99%'
        self.grid.loc[(self.grid['GiPValue'] < 0.05) & (self.grid['GiZScore'] < 0) & (self.grid['hotspot'] == 'Not Significant'), 'hotspot'] = 'Coldspot 95%'
        self.grid.loc[(self.grid['GiPValue'] < 0.10) & (self.grid['GiZScore'] < 0) & (self.grid['hotspot'] == 'Not Significant'), 'hotspot'] = 'Coldspot 90%'

        return self.grid

    def plot_gi_map(self):
        color_dict = {
            'Hotspot 99%': '#800026',
            'Hotspot 95%': '#FC4E2A',
            'Hotspot 90%': '#FD8D3C',
            'Not Significant': '#d9d9d9',
            'Coldspot 90%': '#6baed6',
            'Coldspot 95%': '#3182bd',
            'Coldspot 99%': '#08519c'
        }

        categories = [
            'Hotspot 99%', 'Hotspot 95%', 'Hotspot 90%', 
            'Not Significant', 
            'Coldspot 90%', 'Coldspot 95%', 'Coldspot 99%'
        ]

        grid = self.grid.to_crs(epsg=4326)

        fig, ax = plt.subplots(figsize=(10, 12))
        self.taiwan.to_crs(epsg=4326).plot(ax=ax, color='white', edgecolor='black', linewidth=0.5)

        grid.plot(
            column='hotspot', 
            categorical=True, 
            cmap=mcolors.ListedColormap([color_dict[cat] for cat in categories if cat in grid['hotspot'].unique()]),
            legend=True, 
            edgecolor='white',
            linewidth=0.01, 
            alpha=0.8,
            ax=ax,
            categories=categories,
            legend_kwds={
                'bbox_to_anchor': (1.05, 1),
                'loc': 'upper left',
                'frameon': False
            }
        )

        plt.title('Hotspot Analysis (Getis-Ord Gi*) - 90%, 95%, 99% Confidence Levels')
        plt.axis('off')
        plt.show()

myfont = FontProperties(fname=r"/Users/wangqiqian/Library/Fonts/標楷體.ttf")
sns.set(style="whitegrid", font=myfont.get_name())
plt.rcParams['font.family'] = ['Arial Unicode Ms']

def plot_facility_vs_human_vehicle_subplot(data, facilities, accident_col, accident_type):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, facility_col in enumerate(facilities):

        data['is_human_vehicle'] = (data[accident_col] == accident_type).astype(int)

        grouped = data.groupby(facility_col).agg(
            human_vehicle_ratio=('is_human_vehicle', 'mean'),
            total_count=('is_human_vehicle', 'size')
        ).reset_index()

        sns.barplot(
            data=grouped,
            x=facility_col,
            y='human_vehicle_ratio',
            palette="Blues_d",
            ax=axes[i]
        )

        for index, row in grouped.iterrows():
            axes[i].text(
                x=index, 
                y=row['human_vehicle_ratio'] + 0.001,
                s=f"{round(row['total_count'])}", 
                ha='center', 
                va='bottom', 
                fontsize=10, 
                fontproperties=myfont
            )

        axes[i].set_title(f"Nearby {facility_col}", fontsize=14, fontproperties=myfont)
        axes[i].set_xlabel(f"{facility_col} Count", fontsize=12, fontproperties=myfont)
        axes[i].set_ylabel("Pedestrian-vehicle accident ratio", fontsize=12, fontproperties=myfont)
        axes[i].tick_params(axis='x', labelsize=10)
        axes[i].tick_params(axis='y', labelsize=10)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    for j in range(len(facilities), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def hotspot_in_county(hex, hex_county, countycity_dct, normalize=True, en=True):
     
    if en:
        hex_en = hex.copy()
        hex_county_en = hex_county.copy()

        hex_en['nearest_county'] = hex_en['nearest_county'].map(countycity_dct)
        hex_county_en['COUNTYNAME'] = hex_county_en['COUNTYNAME'].map(countycity_dct)
        hot = hex_en[['hotspot', 'nearest_county']].value_counts()
        count_hexagon_in_county = hex_county_en['COUNTYNAME'].value_counts()

    else:
        hot = hex[['hotspot', 'nearest_county']].value_counts()
        count_hexagon_in_county = hex_county['COUNTYNAME'].value_counts()

    hot_df = hot.reset_index()
    hot_df.columns = ['熱點', '最近縣市', '數量']

    plt.figure(figsize=(12, 6))

    if normalize:
        hot_df['normalized_count'] = hot_df.apply(
            lambda row: row['數量'] / count_hexagon_in_county[row['最近縣市']], axis=1
        )
        sns.barplot(data=hot_df, x='最近縣市', y='normalized_count', hue='熱點', palette='viridis')
        plt.title('Total hotspots in each County/City (Ratio)', fontsize=16)
        plt.ylabel('Ratio', fontsize=12)

    else:
        sns.barplot(data=hot_df, x='最近縣市', y='數量', hue='熱點', palette='viridis')
        plt.title('Total hotspots in each County/City', fontsize=16)
        plt.ylabel('Total', fontsize=12)

    plt.xlabel('Nearest County/City', fontsize=12)
    plt.legend(title='Hotspot', fontsize=10)
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.show()

def attribute_in_city(combined_data, hot, col, countycity_dct, feature_name_map, category_value_map, en=True):

    if en:
        hot_hex = hot.copy()
        hot_hex['nearest_county'] = hot_hex['nearest_county'].map(countycity_dct)
        city_order = [
            'Taipei City', 'New Taipei City',  # Northernmost
            'Taoyuan City', 'Hsinchu City', 'Hsinchu County', 'Yilan County',  # Northern Taiwan
            'Miaoli County', 'Taichung City', 'Changhua County',  # Central Taiwan
            'Chiayi City', 'Chiayi County', 'Tainan City', 'Kaohsiung City', 'Pingtung County',  # Southern Taiwan
            'Hualien County', 'Taitung County'  # Eastern Taiwan
        ]

    else:
        hot_hex = hot.copy()
        city_order = [
            '臺北市', '新北市',  # 最北
            '桃園市', '新竹市', '新竹縣', '宜蘭縣',  # 北部
            '苗栗縣', '臺中市', '彰化縣',  # 中部
            '嘉義市', '嘉義縣', '臺南市', '高雄市', '屏東縣',  # 南部
            '花蓮縣', '臺東縣'  # 東部
        ]

    # 先合併所有熱點 hex 的事故索引與縣市
    city_indices = []
    for city in hot_hex['nearest_county'].unique():
        indices = sum(hot_hex[hot_hex['nearest_county'] == city]['accident_indices'], [])
        city_indices.append((city, indices))

    result = []

    for city, indices in city_indices:
        # project回原始資料
        city_data = combined_data.loc[indices]

        counts = city_data[col].value_counts(normalize=True)  # 計算比例
        for signal_type, ratio in counts.items():
            result.append({'County/City': city, col: signal_type, 'Ratio': ratio})

    result_df = pd.DataFrame(result)

    if en:
        result_df[col] = result_df[col].map(category_value_map[col])
        result_df = result_df.rename(columns=feature_name_map)
        col = feature_name_map.get(col, col)

    # 轉成 pivot table 方便比較
    pivot = result_df.pivot(index='County/City', columns=col, values='Ratio').fillna(0)

    pivot_sorted = pivot.loc[city_order]

    # pivot_sorted = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    pivot_sorted.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab20')

    plt.ylabel('Ratio')
    plt.title(f'Proportions of different {col} within hotspot areas across cities')
    plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    return pivot_sorted


# Not used anymore
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
