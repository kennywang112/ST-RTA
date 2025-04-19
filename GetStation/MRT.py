import os
import json
import pandas as pd

data_lst = [
    'kaoshiung', 'newtaipei', 'taichung',
    'taipei', 'taoyuan', 'trtcmg'
]

os.makedirs("./ComputedData/MRT", exist_ok=True)

def get_data(city):

    url = f"./Data/MRT/{city}.json" # Taichung
    with open(url, "r", encoding="utf-8") as f:
        data_str = f.read()

    data = json.loads(data_str)
    df = pd.DataFrame(data)

    dft = pd.DataFrame()
    dft['ExitName'] = df['ExitName'].apply(lambda x: x['Zh_tw'])
    dft['PositionLat'] = df['ExitPosition'].apply(lambda x: x['PositionLat'])
    dft['PositionLon'] = df['ExitPosition'].apply(lambda x: x['PositionLon'])

    dft.to_csv(f"./ComputedData/MRT/{city}.csv", index=False, encoding='utf-8')

    dft['City'] = city
    return dft

all_dfs = [get_data(city) for city in data_lst]
combined_df = pd.concat(all_dfs, ignore_index=True)

combined_df.to_csv(f"./ComputedData/MRT/full_mrt.csv", index=False, encoding='utf-8')