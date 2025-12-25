import json
import folium
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def scatter_with_regression(grid, x_col, y_col):
    # 計算相關係數
    correlation, _ = pearsonr(grid[x_col], grid[y_col])
    print(f"Pearson correlation coefficient: {correlation:.2f}")

    # 設置圖表風格
    sns.set_theme(style="whitegrid")

    # 創建圖表
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        x=x_col, 
        y=y_col, 
        data=grid, 
        alpha=0.7,  # 散點透明度
        # size=y_col,  # 散點大小根據 y_col 動態調整
        # sizes=(20, 200),  # 散點大小範圍
        hue=y_col,  # 散點顏色根據 y_col 動態調整
        palette="viridis"  # 配色方案
    )

    # 添加回歸線
    sns.regplot(
        x=x_col, 
        y=y_col, 
        data=grid, 
        scatter=False,  # 不繪製散點
        color="red",  # 回歸線顏色
        line_kws={"linewidth": 2},  # 回歸線寬度
        ci=None  # 不顯示置信區間
    )

    # 創建顏色映射
    norm = plt.Normalize(grid[y_col].min(), grid[y_col].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    # 添加顏色條
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation="vertical")
    cbar.set_label(f"{y_col}", fontsize=12)

    # 添加標題和標籤
    plt.title(f"Correlation: {correlation:.2f}", fontsize=18, weight='bold')
    plt.xlabel(x_col, fontsize=14, weight='bold')
    plt.ylabel(y_col, fontsize=14, weight='bold')

    # 顯示圖表
    plt.tight_layout()
    plt.show()

def scatter_with_spearman(grid, x_col, y_col):
    # 計算 Spearman 相關係數
    correlation, _ = spearmanr(grid[x_col], grid[y_col])
    print(f"Spearman correlation coefficient: {correlation:.2f}")

    # 設置圖表風格
    sns.set_theme(style="whitegrid")

    # 創建圖表
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        x=x_col, 
        y=y_col, 
        data=grid, 
        alpha=0.7,  # 散點透明度
        hue=y_col,  # 散點顏色根據 y_col 動態調整
        palette="viridis"  # 配色方案
    )

    # 添加回歸線
    sns.regplot(
        x=x_col, 
        y=y_col, 
        data=grid, 
        scatter=False,  # 不繪製散點
        color="red",  # 回歸線顏色
        line_kws={"linewidth": 2},  # 回歸線寬度
        ci=None  # 不顯示置信區間
    )

    # 創建顏色映射
    norm = mpl.colors.Normalize(vmin=grid[y_col].min(), vmax=grid[y_col].max())
    sm = mpl.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    # 添加顏色條
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation="vertical")
    cbar.set_label(f"{y_col}", fontsize=12)

    # 添加標題和標籤
    plt.title(f"Spearman Correlation: {correlation:.2f}", fontsize=18, weight='bold')
    plt.xlabel(x_col, fontsize=14, weight='bold')
    plt.ylabel(y_col, fontsize=14, weight='bold')

    # 顯示圖表
    plt.tight_layout()
    plt.show()

def calculate(group, feature, target, ratio=False):

    count = len(group[group[feature] == target])

    if ratio:
        total = len(group)
        if total == 0:
            return 0
        return count / total
    else:
        return count
    
def calculate_most_common(group, feature):
    # 計算每個類別的出現次數
    most_common = group[feature].value_counts().idxmax()  # 找到出現次數最多的類別
    return most_common

def calculate_average(group, feature):
    if len(group) > 0:
        return group[feature].mean()
    else:
        return 0

def plot_map_type(data, grid):
    grid = grid.copy()
    grid = grid.drop(columns=['centroid'], errors='ignore')
    grid_json = json.loads(grid.to_json())

    # 地圖中心點
    center = [data['緯度'].mean(), data['經度'].mean()]

    # 英文版底圖
    m = folium.Map(
        location=center, 
        zoom_start=10, 
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri'
    )

    # 定義四種類型的顏色
    def get_color(type_value):
        if type_value == 'Hotspot/Include Infrastructure':
            return "#e44587"  # 紅色 - 熱點且有基礎設施
        elif type_value == 'Hotspot/No Infrastructure':
            return "#e493b5"  # 淺紅色 - 熱點但無基礎設施
        elif type_value == 'Not Significant/Include Infrastructure':
            return "#50e2cf"  # 淺藍色 - 非顯著但有基礎設施
        else:  # 'Not Significant/No Infrastructure'
            return "#b8e2dd"  # 灰色 - 非顯著且無基礎設施

    # 加入格網
    folium.GeoJson(
        grid_json,
        style_function=lambda feature: {
            'fillColor': get_color(feature['properties']['type']),
            'color': 'grey',
            'weight': 0.5,
            'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['type', 'num_accidents', 'full_infrastructure'],
            aliases=['Type:', 'Accidents:', 'Infrastructure:'],
            localize=True
        )
    ).add_to(m)

    # 添加圖例
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 280px; height: 160px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>Legend</h4>
    <p><i class="fa fa-square" style="color:#e44587"></i> Hotspot/Include Infrastructure</p>
    <p><i class="fa fa-square" style="color:#e493b5"></i> Hotspot/No Infrastructure</p>
    <p><i class="fa fa-square" style="color:#50e2cf"></i> Not Significant/Include Infrastructure</p>
    <p><i class="fa fa-square" style="color:#b8e2dd"></i> Not Significant/No Infrastructure</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

