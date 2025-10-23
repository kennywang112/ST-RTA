import numpy as np
import pandas as pd
import bnlearn as bn
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

browser_market_share = {
    'browsers': ['firefox', 'chrome', 'safari', 'edge', 'ie', 'opera'],
    'market_share': [8.61, 69.55, 8.36, 4.12, 2.76, 2.43],
    'color': ['#5A69AF', '#579E65', '#F9C784', '#FC944A', '#F24C00', '#00B825']
}

def get_model(dt, black_list=[], white_list=[]):
    # 學哪些變數之間有邊，結果是一個DAG
    model = bn.structure_learning.fit(dt, methodtype='hc', scoretype='bic', bw_list_method='edges',
                                    # 肇因對於事故類型一定是上游。ex. 不會因為撞路樹而造成患病，而是因為患病才造成撞路樹
                                    black_list=black_list, white_list=white_list,
                                    fixed_edges=white_list, max_indegree=None)
    # 計算每個節點的 條件機率表 (CPT, Conditional Probability Table)
    model_param = bn.parameter_learning.fit(model, dt, scoretype='bdeu', methodtype='bayes')
    # 計算邊緣強度，如果p小於顯著就是有相關
    model_independence = bn.independence_test(model_param, dt, test='chi_square', prune=True)

    return model, model_param, model_independence

# https://matplotlib.org/stable/gallery/misc/packed_bubbles.html
class BubbleChart:
    def __init__(self, area, bubble_spacing=0, text_rotation=0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.text_rotation = text_rotation
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return np.argmin(distance, keepdims=True)

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels, colors):
        """
        Draw the bubble plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : list
            Labels of the bubbles.
        colors : list
            Colors of the bubbles.
        """
        for i in range(len(self.bubbles)):
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
            ax.add_patch(circ)
            ax.text(*self.bubbles[i, :2], labels[i],
                    horizontalalignment='center', verticalalignment='center',
                    rotation=self.text_rotation)

def draw_bn_plotly(model, layout_algo="", en=False, width=1000, height=500, seed=42, iter=100):
    edges = [(str(u), str(v)) for u, v in model['model_edges']]
    df = model['independence_test'][['source','target','p_value']].copy()

    if en:
        df['source'] = df['source'].map(feature_name_map).fillna(df['source'])
        df['target'] = df['target'].map(feature_name_map).fillna(df['target'])
        edges = [(feature_name_map.get(u, u), feature_name_map.get(v, v)) for (u, v) in edges]
    else:
        df['source'] = df['source'].astype(str)
        df['target'] = df['target'].astype(str)

    p_map = {(s,t):p for s,t,p in df.itertuples(index=False, name=None)}
    p_map.update({(t,s):p for (s,t),p in list(p_map.items())})

    G = nx.DiGraph()
    G.add_edges_from(edges)

    pos = (nx.spring_layout(G, seed=seed, iterations=iter) if layout_algo=="spring"
           else nx.kamada_kawai_layout(G))

    # nodes
    deg = dict(G.degree())
    node_x, node_y, node_text, node_size = [], [], [], []
    for n in G.nodes():
        x,y = pos[n]
        node_x.append(x); node_y.append(y)
        node_text.append(f"{n}<br>degree: {deg.get(n,0)}")
        # node_size.append(10 + 25*(deg.get(n,1)))
        node_size.append(50)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=[str(n) for n in G.nodes()],
        textposition="middle center",
        hovertext=node_text, hoverinfo="text",
        marker=dict(size=node_size, 
                    # line=dict(width=1), 
                    line=dict(color="#24475E", width=2),
                    color="#5390B9")
    )

    # edge
    edge_traces = []
    annotations = []
    r = 0.15
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx, dy = x1 - x0, y1 - y0
        d = (dx**2 + dy**2)**0.5
        if d == 0:
            continue

        # 起點：從 source 往 target 方向移動 r
        sx = x0 + dx/d * r
        sy = y0 + dy/d * r
        # 終點：從 target 往 source 方向退 r
        ex = x1 - dx/d * r
        ey = y1 - dy/d * r

        annotations.append(dict(
            ax=sx, ay=sy, x=ex, y=ey,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=2, opacity=0.8
        ))

    fig = go.Figure(data=edge_traces + [node_trace],
        layout=go.Layout(
            template=None, showlegend=False,
            hovermode='closest',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=annotations,
            width=width, height=height,
        )
    )
    return fig

def cpd_add_n(parent, child, model, data, cpd=True, threshold=50):

    if cpd:
        # CPD: P(child | parent) -> counts 也用 parent+child
        vb_all = parent.copy()
        vb_all.append(child)
        counts = (data.groupby(vb_all, dropna=False).size().reset_index(name='n'))

        dfprob_cause_counts = (
            model
            .merge(counts, on=vb_all, how='left')
            .sort_values('p', ascending=False)
        )
    else:
        # Posterior: P(parent | child=v) -> data 已固定 child，counts 只用 parent
        counts = (data.groupby(parent, dropna=False)
                       .size()
                       .reset_index(name='n'))

        dfprob_cause_counts = (
            model
            .merge(counts, on=parent, how='left')
            .sort_values('p', ascending=False)
        )

    dfprob_cause_counts['n'] = dfprob_cause_counts['n'].fillna(0)
    filtered = dfprob_cause_counts[dfprob_cause_counts['n'] >= threshold].copy()

    filtered['p'] = round(filtered['p'], 4)
    filtered['n'] = filtered['n'].astype(int)

    return filtered

def filter_cpd_for_hotspot(filtered):

    filtered = filtered[
        (filtered['速限-第1當事者'] == '0-9') |
        (filtered['速限-第1當事者'] == '10-19') |
        (filtered['速限-第1當事者'] == '20-29') |
        (filtered['速限-第1當事者'] == '30-39') |
        (filtered['速限-第1當事者'] == '40-49') |
        (filtered['速限-第1當事者'] == '50-59')
    ]

    filtered = filtered[filtered['道路類別-第1當事者-名稱'] == '市區道路']

    return filtered

def get_outlier(filtered, new_filtered):
    """
    Q的計算是基於原始的filtered，但是要讓他對比新的filtered
    """

    Q1 = filtered['p'].quantile(0.25)
    Q3 = filtered['p'].quantile(0.75)
    IQR = Q3 - Q1
    outliers_high = new_filtered[new_filtered['p'] > Q3 + 1.5 * IQR]
    outliers_low = new_filtered[new_filtered['p'] < Q1 - 1.5 * IQR]
    outliers_high['type'] = 'high'
    outliers_low['type'] = 'low'
    # outliers = pd.concat([outliers_high , outliers_low], axis=0)
    outliers = pd.concat([outliers_high], axis=0)
    print(outliers)

    return outliers