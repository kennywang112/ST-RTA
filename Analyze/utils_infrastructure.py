import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib as mpl

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
        size=y_col,  # 散點大小根據 y_col 動態調整
        sizes=(20, 200),  # 散點大小範圍
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