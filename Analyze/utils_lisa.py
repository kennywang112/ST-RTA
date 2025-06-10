import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from libpysal.weights import KNN
from esda.moran import Moran_Local

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