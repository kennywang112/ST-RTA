import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

plt.rcParams['font.family'] = ['Arial Unicode Ms']

class MapperPlotterSpring:
    def __init__(self, mapper_info, rbind_data, cmap='jet', seed=10,
                 width=400, height=400, iterations=30, dim=3, range_lst=None):
        """
        mapper_info: networkx.Graph（Mapper 產生的圖）
        rbind_data:  原始資料的 DataFrame（要從這裡用 ids 聚合出每個 node 的顏色）
        range_lst:   [xmin, xmax, ymax, ymin]（和你原本一致）
        """
        self.mapper_info = mapper_info        # nx.Graph
        self.rbind_data = rbind_data
        self.cmap = cmap
        self.iterations = iterations
        self.seed = seed
        self.width = width
        self.height = height
        self.dim = dim
        self.range_lst = range_lst

        # 狀態
        self.G = mapper_info
        self.pos = None
        self.full_info = None
        self.filtered_info = None
        self.outlier_info = None   # 留著介面一致（此版本不計算 outlier）
        self.color_palette = None
        self.unique_categories = None
        self.color_mode_avg = None
        self.encoded_label = None
        self.choose = None

    @staticmethod
    def _pick_ids_attr(attr: dict):
        """嘗試從節點屬性字典抓出原始索引欄位。"""
        for key in ("ids", "members", "membership"):
            if key in attr:
                return key
        return None

    def _aggregate_node_color(self, ids):
        if ids is None or len(ids) == 0:
            return np.nan
        try:
            if isinstance(self.choose, (list, tuple)) and len(self.choose) == 2:
                # 兩欄：拿出該節點的子表（注意這裡用 list(...) 而不是 tuple）
                subdf = self.rbind_data.iloc[list(ids)][list(self.choose)]
                return float(self.encoded_label(subdf))
            else:
                # 單欄：沿用舊行為
                vals = self.rbind_data['color_for_plot'].iloc[list(ids)].tolist()
                return float(self.encoded_label(vals))
        except Exception:
            return np.nan

    def create_mapper_plot(self, choose, encoded_label, avg=False, size_threshold=0):
        print("Creating spring layout...")
        self.choose = choose
        self.color_mode_avg = bool(avg)
        self.encoded_label = encoded_label

        # 只有「單欄」才建立 color_for_plot；雙欄由 _aggregate_node_color 處理
        if not (isinstance(choose, (list, tuple)) and len(choose) == 2):
            if avg:
                self.rbind_data['color_for_plot'] = self.rbind_data[choose].astype(float)
            else:
                self.rbind_data['color_for_plot'] = pd.factorize(self.rbind_data[choose])[0]

        nodes_to_keep = [n for n, attr in self.G.nodes(data=True) if attr.get("size", 0) > size_threshold]
        H = self.G.subgraph(nodes_to_keep).copy()
        self.G = H
        self.pos = nx.spring_layout(self.G, dim=self.dim, seed=self.seed, iterations=self.iterations)
        print(f"Mapper (spring) layout computed. nodes={self.G.number_of_nodes()}")
        return self

    def extract_data(self, rx=False, ry=False, rz=False):
        """
        從 self.G 與 self.pos 組裝 full_info：
        columns = ['node','size','ids','neighbors','color','x','y','(z)','ratio']
        """
        print
        # 鄰居（排序）
        neighbors_map = {n: sorted(list(self.G.neighbors(n))) for n in self.G.nodes()}

        rows = []
        for n, attr in self.G.nodes(data=True):
            # ids / size
            ids_key = self._pick_ids_attr(attr)
            ids = list(attr.get(ids_key, [])) if ids_key is not None else []
            size = attr.get("size", len(ids) if ids else None)

            # 聚合顏色
            color_val = self._aggregate_node_color(ids)

            # 座標
            coords = self.pos[n]
            if self.dim == 3:
                x, y, z = coords[0], coords[1], coords[2]
            else:
                x, y, z = coords[0], coords[1], None

            rows.append({
                "node": n,
                "neighbors": neighbors_map[n],
                "size": size,
                "ids": ids,
                "color": color_val,
                "x": x, "y": y, **({"z": z} if self.dim == 3 else {})
            })

        self.full_info = pd.DataFrame(rows)
        # ratio：若 color 是平均值可當比例；若是分類則會是 NaN
        with np.errstate(invalid='ignore', divide='ignore'):
            self.full_info["ratio"] = self.full_info["color"] / self.full_info["size"]

        # 可選旋轉（3D 時）
        if self.dim == 3 and (rx or ry or rz):
            def _rot_x(P, angle):
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
                return P @ R.T
            def _rot_y(P, angle):
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
                return P @ R.T
            def _rot_z(P, angle):
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
                return P @ R.T

            P = self.full_info[['x','y','z']].to_numpy()
            if rx: P = _rot_x(P, rx)
            if ry: P = _rot_y(P, ry)
            if rz: P = _rot_z(P, rz)
            self.full_info[['x','y','z']] = P

        # 本版本不做 outlier 偵測，保留欄位以維持相容
        self.outlier_info = pd.DataFrame(columns=self.full_info.columns)
        self.filtered_info = self.full_info.copy()
        print("Data extracted.")
        return self.filtered_info, self.outlier_info

    def map_colors(self, choose, size=0, threshold=5, drop_unused_labels=True):
        """
        choose: 要用來顯色的欄位（和 create_mapper_plot 保持一致）
        size:   以節點 size 做最小門檻過濾
        threshold: 只保留在「子圖對應的原始資料子集合」中出現次數 > threshold 的類別
        drop_unused_labels: 若某類別在子圖中完全沒被任何節點代表，則從 legend 與資料中移除
        """
        print("Mapping colors...")

        # 1) 以節點大小過濾 + 依 range_lst 篩座標
        df = self.filtered_info[self.filtered_info['size'] > size].copy()
        if self.range_lst is not None:
            xmin, xmax, ymax, ymin = self.range_lst
            df = df[(df['x'] > xmin) & (df['x'] < xmax) & (df['y'] > ymin) & (df['y'] < ymax)]

        # 如果整個子圖空了，直接收尾
        if df.empty:
            self.filtered_info = df.assign(color_for_plot_fixed=np.nan)
            self.color_palette = {}
            self.unique_categories = []
            print("No nodes remain after filtering.")
            return df

        # 2) 只用子圖所覆蓋到的原始 ids 來決定類別與門檻
        #    這一步是關鍵：不要用全體 rbind_data，而是用子集合 rbind_subset
        ids_series = df['ids'].explode()
        kept_ids = ids_series.dropna().astype(int).unique().tolist()
        rbind_subset = self.rbind_data.iloc[kept_ids].copy()

        if self.color_mode_avg:
            # 連續色：把 color 正規化丟進 cmap；與類別無關
            vals = df['color'].astype(float)
            vmin, vmax = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
            vmax = vmax if vmax > vmin else vmin + 1e-9
            norm = (vals - vmin) / (vmax - vmin)
            cmap = get_cmap(self.cmap)
            df['color_for_plot_fixed'] = norm.apply(lambda t: cmap(np.clip(t, 0, 1)))

            # 連續模式沒有離散圖例
            self.color_palette = {}
            self.unique_categories = []
        else:
            # 離散色：先在「子集合」內計數與門檻
            # 這裡用子集合以免出現未在子圖出現的類別
            category_counts = rbind_subset[self.choose].value_counts(dropna=True)
            filtered_categories = category_counts[category_counts > threshold].index.tolist()

            # 建立子集合的 "編碼 -> 類別名" 對照（仍沿用先前 factorize 的編碼）
            unique_values = (
                rbind_subset.reset_index(drop=True)[[self.choose, 'color_for_plot']]
                .drop_duplicates()
            )
            code_to_cat = dict(zip(unique_values['color_for_plot'], unique_values[self.choose]))

            # 把節點 color（是聚合後的「編碼」）對回「類別名」
            df['_cat'] = df['color'].map(code_to_cat)

            # （可選）丟掉在子圖中根本沒被任何節點代表、或未達門檻的類別
            if drop_unused_labels:
                df = df[df['_cat'].isin(filtered_categories)]

            # 再算一次「子圖中實際出現」的類別清單（legend 用）
            present_cats = df['_cat'].dropna().unique().tolist()

            # 依 filtered_categories 的順序排，圖例會更一致
            ordered_cats = [c for c in filtered_categories if c in present_cats]

            # 產生固定 palette（只對仍存在的類別上色）
            color_palette = get_cmap("tab20", len(ordered_cats))
            color_mapping_fixed = {cat: color_palette(i) for i, cat in enumerate(ordered_cats)}

            # 指派顏色；其他（NaN 或被丟掉的）給預設灰
            default_color = (0.5, 0.5, 0.5, 1)
            df['color_for_plot_fixed'] = df['_cat'].map(color_mapping_fixed).apply(
                lambda c: c if pd.notna(c) else default_color
            )

            # 狀態保存
            self.color_palette = color_mapping_fixed
            self.unique_categories = ordered_cats
            df.drop(columns=['_cat'], inplace=True, errors='ignore')

        self.filtered_info = df
        print("Colors mapped.")
        return df

    def plot(self, choose, avg=None, save_path=None, set_label=False, size=100, anchor=1):
        """
        choose:   顏色欄位名（為相容保留；實際已在 map_colors 用 self.choose）
        avg:      若提供，會覆寫 create_mapper_plot 的 avg 模式（可不給）
        size:     畫圖時節點大小上限（功能 2 的「避免極大值」，用 clip 控制）
        """
        print("Plotting...")
        if avg is not None:
            self.color_mode_avg = bool(avg)

        valid = self.filtered_info.dropna(subset=['color_for_plot_fixed']).copy()
        # clip node size，避免一個節點過大
        clipped_size = np.clip(valid['size'].fillna(1).astype(float), None, size)

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            valid['x'], valid['y'],
            c=[tuple(c) if isinstance(c, (list, tuple)) else c for c in valid['color_for_plot_fixed']],
            edgecolors='black', linewidths=0.5,
            s=clipped_size, marker='o', alpha=0.9
        )

        # 畫邊
        pos_dict = {row['node']:(row['x'], row['y']) for _, row in valid.iterrows()}
        for u, v in self.G.edges():
            if u in pos_dict and v in pos_dict:
                x_coords = [pos_dict[u][0], pos_dict[v][0]]
                y_coords = [pos_dict[u][1], pos_dict[v][1]]
                plt.plot(x_coords, y_coords, color='grey', alpha=0.5, linewidth=0.5, zorder=0)

        if set_label:
            if self.color_mode_avg:
                # 連續色條
                cbar = plt.colorbar(scatter, ax=plt.gca(), orientation='vertical', pad=0.02)
                cbar.set_label(f"{choose}")
            else:
                # 離散 legend
                handles = [
                    plt.Line2D([0],[0], marker='o', color=self.color_palette[name],
                               markersize=10, linestyle='None', label=name)
                    for name in (self.unique_categories or [])
                ]
                plt.legend(handles=handles, title=f"{choose}", loc='upper right', bbox_to_anchor=(anchor, 1))

        plt.xlabel('X'); plt.ylabel('Y'); plt.title('Mapper (spring) plot'); plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plot3d_matplotlib(self, choose, avg=None, save_path=None, size=1200,
                        elev=22, azim=35, dpi=150):
        """
        3D 靜態作圖（Matplotlib）
        - 需先 create_mapper_plot(..., dim=3) + extract_data() + map_colors()
        - 不用 HTML，不依賴 plotly/kaleido
        - size: 節點大小上限（會 clip）
        - elev/azim: 視角設定
        - dpi: 圖面解析度（存檔 & 顯示）
        """
        if self.dim != 3:
            raise ValueError("plot3d_matplotlib 需要 dim=3 才能使用。")

        if avg is not None:
            self.color_mode_avg = bool(avg)

        df = self.filtered_info.dropna(subset=['x','y','z']).copy()
        if df.empty:
            raise ValueError("filtered_info 為空，請先呼叫 map_colors() 產生顏色並完成篩選。")

        # 節點大小：clip 後略做根號縮放，讓視覺更均衡
        sizes = np.clip(df['size'].fillna(1).astype(float), None, size)
        sizes = np.sqrt(sizes)  # 3D 視覺上更平衡

        # 顏色: 你在 map_colors() 已經產出 RGBA，可以直接用
        colors = [
            tuple(c) if isinstance(c, (list, tuple)) else (0.5, 0.5, 0.5, 1.0)
            for c in df['color_for_plot_fixed']
        ]

        # 建立 3D 畫布
        fig = plt.figure(figsize=(12, 10), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)

        # 畫節點
        sc = ax.scatter(
            df['x'], df['y'], df['z'],
            s=sizes,
            c=colors,
            edgecolors='k',
            linewidths=0.4,
            alpha=0.95
        )

        # 畫邊
        pos = {row['node']: (row['x'], row['y'], row['z']) for _, row in df.iterrows()}
        for u, v in self.G.edges():
            if u in pos and v in pos:
                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]
                ax.plot([x0, x1], [y0, y1], [z0, z1],
                        color='grey', alpha=0.45, linewidth=0.7, zorder=0)

        # 軸與標題
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('Mapper (spring) 3D', pad=16)
        # 清爽一點
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]['linewidth'] = 0.3
            axis._axinfo["grid"]['color'] = (0.8, 0.8, 0.8, 0.5)

        # Legend（離散類別時）
        if not self.color_mode_avg and self.unique_categories:
            handles = []
            for name in self.unique_categories:
                c = self.color_palette.get(name, (0.5, 0.5, 0.5, 1))
                handles.append(plt.Line2D([0],[0], marker='o', linestyle='None',
                                        markerfacecolor=c, markeredgecolor='k',
                                        markersize=8, label=str(name)))
            ax.legend(handles=handles, title=str(choose), loc='upper left', bbox_to_anchor=(1.02, 1.0))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"3D plot saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()