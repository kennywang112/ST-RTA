import json
import pandas as pd

data_lst = [
    'kaoshiung', 'newtaipei', 'taichung',
    'taipei', 'taoyuan', 'trtcmg'
]

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

for city in data_lst:
    get_data(city)