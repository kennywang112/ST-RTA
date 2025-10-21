import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

mrt = pd.read_csv('./ComputedData/MRT/full_mrt.csv')
youbike = pd.read_csv('./ComputedData/YouBike/full_youbike.csv')
parkinglot = pd.read_csv('./ComputedData/Parkinglot/full_parkinglot.csv')

dataA2 = pd.read_csv("./Data/Accident/A2.csv", low_memory=False)
dataA1 = pd.read_csv("./Data/Accident/A1.csv")

facilities = {
    'mrt': mrt,
    'youbike': youbike,
    'parkinglot': parkinglot
}

# os.makedirs("./ComputedData/Accident", exist_ok=True)

def Calculate(X, facility_dict, name):

    # Step 1: 轉GeoDataFrame（經度、緯度到幾何點）投影為平面坐標系
    X['geometry'] = [Point(xy) for xy in zip(X['經度'], X['緯度'])]
    gdf_data = gpd.GeoDataFrame(X, geometry='geometry', crs="EPSG:4326").to_crs(epsg=3826)

    # Step 2: 建立500公尺的範圍
    gdf_data['buffer'] = gdf_data.geometry.buffer(100)
    gdf_buffer = gdf_data.set_geometry('buffer')

    # Step 3: 每個設施資料逐一處理
    for label, facility in facility_dict.items():

        facility['geometry'] = [Point(xy) for xy in zip(facility['PositionLon'], facility['PositionLat'])]
        gdf_facility = gpd.GeoDataFrame(facility, geometry='geometry', crs="EPSG:4326").to_crs(epsg=3826)

        # 空間join
        joined = gpd.sjoin(gdf_buffer, gdf_facility, how='left', predicate='intersects')
        joined['index_left'] = joined.index
        valid = joined[~joined['index_right'].isna()]
        counts = valid.groupby('index_left').size().reindex(gdf_buffer.index, fill_value=0)

        # 新增欄位: 該設施在 100 公尺內的數量
        gdf_data[f'{label}_100m_count'] = gdf_data.index.map(counts).fillna(0).astype(int)

    # Step 4: 清理和儲存
    gdf_data.drop(columns=['geometry', 'buffer'], inplace=True)
    gdf_data.to_csv(f'./ComputedDataV2/Accident/{name}.csv', index=False, encoding='utf-8')

Calculate(dataA1, facilities, 'DataA1_with_MYP')
Calculate(dataA2, facilities, 'DataA2_with_MYP')