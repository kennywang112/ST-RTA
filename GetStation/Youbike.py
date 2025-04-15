import json
import pandas as pd

data_lst = [
    'changhua', 'chiayi', 'chiayicountry', 'hsinchu', 'hsinchucountry',
    'kaoshiung', 'miaoli', 'newtaipei', 'pingtung', 'taichung',
    'tainan', 'taipei', 'taitung', 'taoyuan', 'yunlin'
]

def get_data(city):

    url = f"./Data/Youbike/{city}.json"
    with open(url, "r", encoding="utf-8") as f:
        data_str = f.read()

    data = json.loads(data_str)
    df = pd.DataFrame(data)

    dft = pd.DataFrame()
    dft['StationName'] = df['StationName'].apply(lambda x: x['Zh_tw'])
    dft['PositionLat'] = df['StationPosition'].apply(lambda x: x['PositionLat'])
    dft['PositionLon'] = df['StationPosition'].apply(lambda x: x['PositionLon'])

    dft.to_csv(f"./ComputedData/Youbike/{city}.csv", index=False, encoding='utf-8')

for city in data_lst:
    print(f"Start {city}")
    get_data(city)