import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

plt.rcParams['font.family'] = ['Arial Unicode Ms']

class MapperPlotterSpring:
    def __init__(self, mapper_info, rbind_data, cmap='jet', seed=10,
                 width=400, height=400, iterations=30, dim=3, range_lst=None,
                 encoded_label=None):
        """
        mapper_info: networkx.Graph(Mapper 產生的圖)
        rbind_data: 原始資料的 DataFrame(要從這裡用 ids 聚合出每個 node 的顏色)
        range_lst: [xmin, xmax, ymax, ymin](和你原本一致)
        encoded_label: node_cond_prob, most_common_encoded_label, ratio_in_data, avg_label
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
        self.encoded_label = encoded_label

        # 狀態
        self.G = mapper_info
        self.pos = None
        self.full_info = None
        self.filtered_info = None
        self.outlier_info = None   # 留著介面一致（此版本不計算 outlier）
        self.color_palette = None
        self.unique_categories = None
        self.color_mode_avg = None
        self.choose = None
        self.size_threshold = None

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

    def _rgba_to_hex(c):
        """(r,g,b,a) or (r,g,b) -> '#RRGGBB'；若已是 hex 或字串就原樣返回。"""
        import matplotlib.colors as mcolors
        if c is None or (isinstance(c, float) and np.isnan(c)):
            return "#808080"
        if isinstance(c, str):
            # 已是顏色字串：可能是 '#RRGGBB' 或 'rgb(...)' 或色名
            try:
                return mcolors.to_hex(c, keep_alpha=False)
            except Exception:
                return "#808080"
        try:
            return mcolors.to_hex(c, keep_alpha=False)
        except Exception:
            return "#808080"

    def create_mapper_plot(self, choose, avg=False, size_threshold=0, plot_type='spring'):

        print("Creating spring layout...")
        self.size_threshold = size_threshold
        self.choose = choose
        self.color_mode_avg = bool(avg)

        if not (isinstance(choose, (list, tuple)) and len(choose) == 2):
            col = self.rbind_data[choose]
            if avg:
                # 連續模式：只有數值欄位才轉成 float；非數值保留原樣，讓 encoded_label 自行聚合
                if pd.api.types.is_numeric_dtype(col):
                    self.rbind_data['color_for_plot'] = pd.to_numeric(col, errors='coerce')
                else:
                    self.rbind_data['color_for_plot'] = col.astype(str)
            else:
                # 離散模式：factorize
                self.rbind_data['color_for_plot'] = pd.factorize(col)[0]

        # 篩選節點
        nodes_to_keep = [n for n, attr in self.G.nodes(data=True) if attr.get("size", 0) > self.size_threshold]
        H = self.G.subgraph(nodes_to_keep).copy()
        self.G = H

        if plot_type == 'spring':
            self.pos = nx.spring_layout(self.G, dim=self.dim, seed=self.seed, iterations=self.iterations)
        else:
            self.pos = nx.kamada_kawai_layout(self.G, dim=self.dim)

        print(f"Mapper layout computed. nodes={self.G.number_of_nodes()}")
        return self

    def extract_data(self, rx=False, ry=False, rz=False):
        """
        從 self.G 與 self.pos 組裝 full_info:
        columns = ['node','size','ids','neighbors','color','x','y','(z)','ratio']
        """

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
                "x": x,
                "y": y,
                **({"z": z} if self.dim == 3 else {})
            })

        self.full_info = pd.DataFrame(rows)
        # ratio：若 color 是平均值可當比例；若是分類則會是 NaN
        with np.errstate(invalid='ignore', divide='ignore'):
            self.full_info["ratio"] = self.full_info["color"] / self.full_info["size"]

        # 3D時可旋轉
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

        # 目前不做find_connected_points找 outlier
        self.outlier_info = pd.DataFrame(columns=self.full_info.columns)
        self.filtered_info = self.full_info.copy()
        print("Data extracted.")
        return self.filtered_info, self.outlier_info

    def map_colors(self, threshold=5, drop_unused_labels=True, en={}):
        """
        choose: 要用來顯色的欄位（和 create_mapper_plot 保持一致）
        threshold: 只保留在「子圖對應的原始資料子集合」中出現次數 > threshold 的類別
        drop_unused_labels: 若某類別在子圖中完全沒被任何節點代表，則從 legend 與資料中移除
        """
        print("Mapping colors...")

        # 以節點大小過濾 + 依 range_lst 篩座標
        df = self.filtered_info[self.filtered_info['size'] > self.size_threshold].copy()
        if self.range_lst is not None:
            xmin, xmax, ymax, ymin = self.range_lst
            df = df[(df['x'] > xmin) & (df['x'] < xmax) & (df['y'] > ymin) & (df['y'] < ymax)]

        # 只用子圖所覆蓋到的原始 ids 來決定類別與門檻
        # 不要用全體 rbind_data，而是用子集合 rbind_subset
        ids_series = df['ids'].explode()
        kept_ids = ids_series.dropna().astype(int).unique().tolist()
        rbind_subset = self.rbind_data.iloc[kept_ids].copy()

        if self.color_mode_avg:
            # 連續色：把 color 正規化丟進 cmap；與類別無關
            vals = df['color'].astype(float)
            # 2, 98 作為映射範圍，避免極端值影響
            vmin, vmax = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
            vmax = vmax if vmax > vmin else vmin + 1e-9
            # standardization
            norm = (vals - vmin) / (vmax - vmin)
            cmap = cm.get_cmap(self.cmap)
            df['color_for_plot_fixed'] = norm.apply(lambda t: cmap(np.clip(t, 0, 1)))

            # 連續模式沒有離散圖例
            self.color_palette = {}
            self.unique_categories = []
            self._cont_vmin = float(vmin)
            self._cont_vmax = float(vmax)
            self._cont_cmap = self.cmap
        else:
            # 離散色：先在「子集合」內計數與門檻
            # 這裡用子集合以免出現未在子圖出現的類別
            category_counts = rbind_subset[self.choose].value_counts(dropna=True)
            filtered_categories = category_counts[category_counts > threshold].index.tolist()

            # 建立子集合的 "編碼 -> 類別名" 對照
            unique_values = (
                rbind_subset.reset_index(drop=True)[[self.choose, 'color_for_plot']]
                .drop_duplicates()
            )
            code_to_cat = dict(zip(unique_values['color_for_plot'], unique_values[self.choose]))

            # 把節點 color（是聚合後的「編碼」）對回「類別名」
            df['_cat'] = df['color'].map(code_to_cat)

            # 丟掉在子圖中根本沒被任何節點代表、或未達門檻的類別
            if drop_unused_labels:
                df = df[df['_cat'].isin(filtered_categories)]

            # 再算一次「子圖中實際出現」的類別清單（legend 用）
            present_cats = df['_cat'].dropna().unique().tolist()

            # 依 filtered_categories 的順序排，圖例會更一致
            # ordered_cats = [c for c in filtered_categories if c in present_cats]
            ordered_cats_cn = [c for c in filtered_categories if c in present_cats]

            df['_cat'] = df['_cat'].map(en).fillna(df['_cat'])
            ordered_cats = [en.get(c, c) for c in ordered_cats_cn]

            # 產生固定 palette（只對仍存在的類別上色）
            color_palette = cm.get_cmap("tab20", len(ordered_cats))
            color_mapping_fixed = {cat: color_palette(i) for i, cat in enumerate(ordered_cats)}

            # 指派顏色；其他（NaN 或被丟掉的）給預設灰
            default_color = (0.5, 0.5, 0.5, 1)
            df['color_for_plot_fixed'] = df['_cat'].map(color_mapping_fixed).apply(
                lambda c: c if pd.notna(c) else default_color
            )

            self.color_palette = color_mapping_fixed
            self.unique_categories = ordered_cats
            df.drop(columns=['_cat'], inplace=True, errors='ignore')
            self._cont_vmin = None
            self._cont_vmax = None
            self._cont_cmap = None

        self.filtered_info = df
        print("Colors mapped.")
        return df

    def plot(self, save_path=None, set_label=False, size=100, anchor=1):
        """
        size: 畫圖時節點大小上限（功能 2 的「避免極大值」，用 clip 控制）
        """
        print("Plotting...")
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
                vmin = getattr(self, "_cont_vmin", np.nanmin(valid["color"]))
                vmax = getattr(self, "_cont_vmax", np.nanmax(valid["color"]))
                cmap = plt.get_cmap(getattr(self, "_cont_cmap", self.cmap))
                mappable = cm.ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
                mappable.set_array([])  # 避免 warning
                cbar = plt.colorbar(mappable, ax=plt.gca(), orientation='vertical', pad=0.02)
                cbar.set_label(f"{self.choose}")
            else:
                # 離散 legend
                handles = [
                    plt.Line2D([0],[0], marker='o', color=self.color_palette[name],
                               markersize=10, linestyle='None', label=name)
                    for name in (self.unique_categories or [])
                ]
                plt.legend(handles=handles, title=f"{self.choose}", loc='lower left', bbox_to_anchor=anchor)

        plt.xlabel('X'); plt.ylabel('Y'); plt.title('Mapper (spring) plot'); plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def pvis(self,
            path: str = "mapper_pvis.html",
            height: str = "800px",
            width: str = "100%",
            notebook: bool = False,
            open_browser: bool = False,
            physics_mode: str = "warm",  # "off" | "warm"
            node_size_cap: float = 100.0,
            scale_xy: float = 800.0,
            tooltip_fields: tuple = ("size", "ratio"),
            edge_length_from_layout: bool = True,   # << 新增開關
            ):
        """
        產生 PyVis 互動網頁。
        需求：你已經跑過 create_mapper_plot()、extract_data()、map_colors()

        path: 輸出的 HTML 檔名
        physics_mode: "off" -> 完全關閉物理; "warm" -> 溫和物理，不重跑穩定化
        node_size_cap: 節點大小上限 (像素)
        scale_xy: spring_layout 座標的縮放倍率
        tooltip_fields: 滑鼠提示顯示的欄位
        edge_length_from_layout: 若 True，每條邊的長度設為 spring_layout 的距離
        """
        from pyvis.network import Network
        import numpy as np
        import pandas as pd

        df = self.filtered_info.dropna(subset=["x", "y"]).copy()

        # 顏色
        if "color_for_plot_fixed" not in df.columns:
            if "color" in df.columns:
                df["color_for_plot_fixed"] = df["color"].apply(self._rgba_to_hex)
            else:
                df["color_for_plot_fixed"] = "#808080"
        else:
            df["color_for_plot_fixed"] = df["color_for_plot_fixed"].apply(self._rgba_to_hex)

        net = Network(height=height, width=width, notebook=notebook, directed=False)

        if physics_mode == "off":
            net.set_options("""{
            "physics": { "enabled": false },
            "interaction": { "hover": true, "dragNodes": true, "dragView": true, "zoomView": true },
            "layout": { "improvedLayout": false },
            "nodes": { 
                            "borderWidth": 1,
                            "scaling": { "min": 1, "max": 5 } 
                            },
            "configure": { "enabled": true, "filter": ["physics"] }
            }""")

        elif physics_mode == "warm":
            net.set_options("""{
            "physics": {
                "enabled": true,
                "stabilization": { "enabled": false },
                "solver": "forceAtlas2Based",
                "maxVelocity": 5,
                "minVelocity": 0.1,
                "timestep": 0.35,
                "forceAtlas2Based": {
                "gravitationalConstant": -3,
                "springLength": 1,
                "springConstant": 0.06,
                "damping": 0.9,
                "avoidOverlap": 0.7
                }
            },
            "layout": { "improvedLayout": false },
            "interaction": { "hover": true, "dragNodes": true, "dragView": true, "zoomView": true },
            "nodes": { "borderWidth": 1 },
            "configure": { "enabled": true, "filter": ["physics"] }
            }""")

        present_nodes = set(df["node"].tolist())
        pos_px = {}  # 存像素座標，方便算邊長
        for _, row in df.iterrows():
            nid = int(row["node"]) if pd.notna(row["node"]) else row["node"]

            val = float(row.get("size", 1.0) or 1.0)
            val = float(np.clip(val, 1.0, node_size_cap))

            # tooltip
            title_parts = [f"<b>node</b>: {nid}"]
            if "neighbors" in row and isinstance(row["neighbors"], (list, tuple)):
                title_parts.append(f"<b>neighbors</b>: {len(row['neighbors'])}")
            for key in tooltip_fields:
                if key in row and pd.notna(row[key]):
                    title_parts.append(f"<b>{key}</b>: {row[key]}")
            if "ids" in row and isinstance(row["ids"], (list, tuple)):
                title_parts.append(f"<b>#ids</b>: {len(row['ids'])}")
            title_html = "<br>".join(title_parts)

            # spring_layout 座標轉像素
            x = float(row["x"]) * scale_xy
            y = float(row["y"]) * scale_xy
            pos_px[nid] = (x, y)

            net.add_node(
                n_id=nid,
                title=title_html,
                color=row["color_for_plot_fixed"],
                size=val,
                value=val,
                x=x, y=y
            )
        for u, v in self.G.edges():
            if (u in present_nodes) and (v in present_nodes):
                if edge_length_from_layout:
                    # 用目前 layout 的距離當邊長
                    x1, y1 = pos_px[int(u)]
                    x2, y2 = pos_px[int(v)]
                    dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    net.add_edge(int(u), int(v), length=max(1.0, float(dist)))
                else:
                    net.add_edge(int(u), int(v))

        net.write_html(path, notebook=notebook, open_browser=open_browser)
        print(f"Success: {path}")
        return path

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

# mapper_plotter = MapperPlotterSpring(
#     detailed_results_df['mapper_info'],
#     all_features_df,
#     seed=seed, iterations=130, dim=2,
#     range_lst=[-0.5, 0.5, 0.5, -0.5],
#     cmap="Reds",
#     # encoded_label=node_cond_prob
#     # encoded_label=most_common_encoded_label
#     encoded_label=ratio_in_data
#     # encoded_label=avg_label
# )
# mapper_plotter.create_mapper_plot(choose, avg=True, size_threshold=50, plot_type='spring')
# full_info, outliers = mapper_plotter.extract_data()
# mapper_plotter.map_colors(threshold=0)
# mapper_plotter.plot(set_label=True, size=500, anchor=(0,0),
#                     save_path=f"../ComputedData/ForMatrixV2/Plots/o{overlap}i{interval}s{seed}_{choose}.png"
#                     )
# mapper_plotter.plot3d_matplotlib(
#     choose,
#     avg=False,
#     save_path=f"../ComputedData/ForMatrixV2/Plots/o{overlap}i{interval}s{seed}_{choose}.png",
#     size=1200, elev=22, azim=35, dpi=180
# )
# mapper_plotter.pvis(
#     path="mapper_interactive.html",
#     physics_mode="off",
#     edge_length_from_layout=True,
#     scale_xy=900,
#     node_size_cap=20,
#     open_browser=True
# )