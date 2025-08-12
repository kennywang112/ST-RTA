import os
import json
import pandas as pd

data_lst = [
    "changhuacountry", "chiayi", "chiayicountry",
    "hsinchu", "hsinchucountry", "hualiencountry",
    "kaohsiung", "keelung", "miaolicountry",
    "nantoucountry", "pingtungcountry", "taichung",
    "tainan", "taipei", "taitungcountry",
    "taoyuan", "yilancountry", "yunlincountry"
]

os.makedirs("./ComputedData/Parkinglot", exist_ok=True)

def get_data(city):

    url = f"./Data/Parkinglot/{city}.json" # Taichung
    with open(url, "r", encoding="utf-8") as f:
        data_str = f.read()

    data = json.loads(data_str)
    df = pd.DataFrame(data['CarParks'])

    dft = pd.DataFrame()
    dft['ExitName'] = df['CarParkName'].apply(lambda x: x['Zh_tw'])
    dft['PositionLat'] = df['CarParkPosition'].apply(lambda x: x['PositionLat'])
    dft['PositionLon'] = df['CarParkPosition'].apply(lambda x: x['PositionLon'])

    dft.to_csv(f"./ComputedData/Parkinglot/{city}.csv", index=False, encoding='utf-8')

    dft['City'] = city
    return dft

all_dfs = [get_data(city) for city in data_lst]
combined_df = pd.concat(all_dfs, ignore_index=True)

combined_df.to_csv(f"./ComputedData/Parkinglot/full_parkinglot.csv", index=False, encoding='utf-8')