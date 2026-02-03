import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, box
from shapely import wkt
from esda.getisord import G_Local
import libpysal

class TaiwanBaseGridGenerator:
    def __init__(self, osm_dir, boundary_shp_path):
        self.osm_dir = osm_dir
        self.boundary_shp_path = boundary_shp_path
        self.crs_wgs84 = "EPSG:4326"
        self.crs_twd97 = "EPSG:3826"
        self.grid = None

    def _load_boundary(self, clip_bounds=None):
        gdf = gpd.read_file(self.boundary_shp_path)
        exclude_towns = ['蘭嶼鄉', '綠島鄉', '琉球鄉']
        exclude_counties = ['金門縣', '連江縣', '澎湖縣']
        
        mask = (~gdf['TOWNNAME'].isin(exclude_towns)) & \
               (~gdf['COUNTYNAME'].isin(exclude_counties))
        taiwan_mainland = gdf[mask].copy()

        if taiwan_mainland.crs != self.crs_twd97:
            taiwan_mainland = taiwan_mainland.to_crs(self.crs_twd97)
            
        if clip_bounds is None:
            bounds = taiwan_mainland.total_bounds
            clip_box = box(bounds[0], 2400000, 380000, bounds[3]) 
        else:
            clip_box = box(*clip_bounds)

        taiwan_clipped = gpd.clip(taiwan_mainland, clip_box)
        return taiwan_clipped

    def generate_grid(self, radius_meters=100, clip_bounds=None):
        print("Loading Map")
        gdf_boundary = self._load_boundary(clip_bounds)
        
        print("Generating Grid")
        minx, miny, maxx, maxy = gdf_boundary.total_bounds

        w = 2 * radius_meters
        h = np.sqrt(3) * radius_meters
        h_step = 1.5 * radius_meters
        v_step = h

        x_coords = np.arange(minx, maxx + w, h_step)
        y_coords = np.arange(miny, maxy + h, v_step)
        
        cols_indices = np.arange(len(x_coords))
        xx, yy = np.meshgrid(x_coords, y_coords)

        xx_cols = np.tile(cols_indices, (len(y_coords), 1))
        
        yy = yy + (xx_cols % 2) * (h / 2)
        x_flat = xx.flatten()
        y_flat = yy.flatten()

        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        hex_offsets_x = radius_meters * np.cos(angles)
        hex_offsets_y = radius_meters * np.sin(angles)
        
        polygons = []
        for cx, cy in zip(x_flat, y_flat):
            poly_coords = np.column_stack((cx + hex_offsets_x, cy + hex_offsets_y))
            polygons.append(Polygon(poly_coords))

        grid_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=self.crs_twd97)

        joined = gpd.sjoin(grid_gdf, gdf_boundary[['geometry']], how='inner', predicate='intersects')
        valid_grid = joined[~joined.index.duplicated(keep='first')].copy()
        self.grid = valid_grid.drop(columns=['index_right']).reset_index(drop=True)
        
        print(f"Grid created with {len(self.grid)} hexagons")
        return self.grid

    def calculate_osm_features(self):
        if self.grid is None:
            raise ValueError("Grid not generated yet.")

        hex_grid = self.grid.copy()
        if 'grid_id' not in hex_grid.columns:
            hex_grid['grid_id'] = range(len(hex_grid))
            
        features_df = pd.DataFrame(index=hex_grid['grid_id'])
        bbox_4326 = hex_grid.to_crs(self.crs_wgs84).total_bounds
        
        roads_path = f"{self.osm_dir}gis_osm_roads_free_1.shp"
        roads = gpd.read_file(roads_path, bbox=tuple(bbox_4326))
        roads = roads.to_crs(self.crs_twd97)
        
        roads_clipped = gpd.overlay(roads, hex_grid[['grid_id', 'geometry']], how='intersection')
        roads_clipped['length_m'] = roads_clipped.length 
        
        road_stats = roads_clipped.groupby(['grid_id', 'fclass'])['length_m'].sum().unstack(fill_value=0)
        road_stats.columns = [f'road_len_{col}' for col in road_stats.columns]
        features_df = features_df.join(road_stats, how='left').fillna(0)

        point_tasks = [
            ("gis_osm_traffic_free_1.shp", "fclass", {
                "traffic_signals": "count_traffic_signals", 
                "stop": "count_stop",
                "crossing": "count_crossing",
                "motorway_junction": "count_motorway_junction",
                "speed_camera": "count_speed_camera",
                "parking": "count_parking"
            }),
            ("gis_osm_transport_free_1.shp", "fclass", {
                "bus_stop": "count_bus_stop",
                "railway_station": "count_train_station"
            })
            # ("gis_osm_pois_free_1.shp", "fclass", {
            #     "pub": "count_alcohol",   
            #     "bar": "count_alcohol",
            #     "nightclub": "count_alcohol",
            #     "convenience": "count_convenience",
            #     "school": "count_school", 
            #     "kindergarten": "count_school"
            # })
        ]

        for filename, filter_col, mapping in point_tasks:
            try:
                gdf_points = gpd.read_file(f"{self.osm_dir}{filename}", bbox=tuple(bbox_4326))
            except Exception:
                continue
                
            gdf_points = gdf_points.to_crs(self.crs_twd97)
            target_values = mapping.keys()
            gdf_points = gdf_points[gdf_points[filter_col].isin(target_values)].copy()
            gdf_points['category'] = gdf_points[filter_col].map(mapping)

            joined = gpd.sjoin(gdf_points, hex_grid[['grid_id', 'geometry']], how='inner', predicate='within')
            counts = joined.groupby(['grid_id', 'category']).size().unstack(fill_value=0)
            
            for col in counts.columns:
                if col in features_df.columns:
                    features_df[col] = features_df[col].add(counts[col], fill_value=0)
                else:
                    features_df = features_df.join(counts[[col]], how='left').fillna(0)

        self.grid = hex_grid.merge(features_df, left_on='grid_id', right_index=True)
        return self.grid

    def add_local_features(self, tasks_dict):
        updated_grid = self.grid.copy()

        for col_name, file_path in tasks_dict.items():
            print(f"正在處理 {col_name} ({file_path})...")
            try:
                df = pd.read_csv(file_path)
                gdf_points = gpd.GeoDataFrame(
                    df, 
                    geometry=gpd.points_from_xy(df['PositionLon'], df['PositionLat']),
                    crs=self.crs_wgs84
                )
                
                gdf_points = gdf_points.to_crs(updated_grid.crs)
                joined = gpd.sjoin(gdf_points, updated_grid[['grid_id', 'geometry']], how='inner', predicate='within')

                counts = joined.groupby('grid_id').size()
                counts.name = col_name

                updated_grid = updated_grid.merge(counts, on='grid_id', how='left')
                updated_grid[col_name] = updated_grid[col_name].fillna(0).astype(int)
            except Exception as e:
                print(f"   [錯誤] 無法處理 {col_name}: {e}")
                updated_grid[col_name] = 0
        
        self.grid = updated_grid
        return self.grid

    def save_base_grid(self, output_path):
        self.grid.to_csv(output_path, index=False)

class AccidentHotspotAnalyzer:
    def __init__(self, base_grid_path):
        self.crs_wgs84 = "EPSG:4326"
        self.crs_twd97 = "EPSG:3826"
        
        print("Loading Base Grid...")
        df = pd.read_csv(base_grid_path)
        df['geometry'] = df['geometry'].apply(wkt.loads)
        self.grid = gpd.GeoDataFrame(df, geometry='geometry', crs=self.crs_twd97)
        print(f"Base Grid loaded with {len(self.grid)} hexagons.")

    def integrate_accident_data(self, accident_csv_path, filter_query=None):
        combined_data = pd.read_csv(accident_csv_path)

        if filter_query:
            print(f"Filtering accidents with query: {filter_query}")
            combined_data = combined_data.query(filter_query)
        else:
            print("Using all accident data.")
            
        gdf_accidents = gpd.GeoDataFrame(
            combined_data,
            geometry=gpd.points_from_xy(combined_data['經度'], combined_data['緯度']),
            crs=self.crs_wgs84
        )

        if gdf_accidents.crs != self.crs_twd97:
            gdf_accidents = gdf_accidents.to_crs(self.crs_twd97)

        print(f"事故點數量 (Filtered): {len(gdf_accidents)}")

        # [關鍵修正]：強制移除 index_right，避免 sjoin 報錯
        if 'index_right' in gdf_accidents.columns:
            gdf_accidents = gdf_accidents.drop(columns=['index_right'])
            
        if 'index_right' in self.grid.columns:
            self.grid = self.grid.drop(columns=['index_right'])

        # 清理舊的分析結果
        cols_to_drop = ['accident_count', 'gi_z', 'gi_p', 'gi_category']
        self.grid = self.grid.drop(columns=[c for c in cols_to_drop if c in self.grid.columns])

        # 進行空間連結
        joined = gpd.sjoin(gdf_accidents, self.grid[['grid_id', 'geometry']], how='inner', predicate='within')
        
        accident_counts = joined.groupby('grid_id').size()
        accident_counts.name = 'accident_count'
        
        self.grid = self.grid.merge(accident_counts, on='grid_id', how='left')
        self.grid['accident_count'] = self.grid['accident_count'].fillna(0).astype(int)

        return self.grid

    def calculate_hotspots(self, target_col='accident_count'):
        work_gdf = self.grid.copy()
        work_gdf[target_col] = work_gdf[target_col].fillna(0)

        w = libpysal.weights.Queen.from_dataframe(work_gdf, use_index=True)

        if w.islands:
            work_gdf = work_gdf.drop(index=w.islands)
            w = libpysal.weights.Queen.from_dataframe(work_gdf, use_index=True)

        w = libpysal.weights.util.fill_diagonal(w, 1)

        y = work_gdf[target_col].values.astype(float)
        g_local = G_Local(y, w, transform='R', star=None)

        work_gdf['gi_z'] = g_local.Zs
        work_gdf['gi_p'] = g_local.p_sim

        conditions = [
            (work_gdf['gi_p'] < 0.01) & (work_gdf['gi_z'] > 0),
            (work_gdf['gi_p'] < 0.05) & (work_gdf['gi_z'] > 0),
            (work_gdf['gi_p'] < 0.10) & (work_gdf['gi_z'] > 0),
            (work_gdf['gi_p'] < 0.01) & (work_gdf['gi_z'] < 0),
            (work_gdf['gi_p'] < 0.05) & (work_gdf['gi_z'] < 0),
            (work_gdf['gi_p'] < 0.10) & (work_gdf['gi_z'] < 0),
        ]
        choices = [
            'Hot Spot (99%)', 'Hot Spot (95%)', 'Hot Spot (90%)',
            'Cold Spot (99%)', 'Cold Spot (95%)', 'Cold Spot (90%)'
        ]
        work_gdf['gi_category'] = np.select(conditions, choices, default='Not Significant')
        
        self.grid = work_gdf
        return self.grid

    def save_result(self, output_path):
        self.grid.to_csv(output_path, index=False)