import json
import requests
import pandas as pd

url_tp = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json" # Taipei

response = requests.get(url_tp)
data_tp = response.json()
df_tp = pd.DataFrame(data_tp)

df_tp.to_csv("./ComputedData/youbike_Taipei.csv", index=False, encoding='utf-8')


url_tc = "./Data/臺中市公共自行車(YouBike2.0)租借站&即時車位資料.json" # Taichung

with open(url_tc, "r", encoding="utf-8") as f:
    data_tc_str = f.read()
data = json.loads(data_tc_str)
df_tc = pd.DataFrame(data['retVal'])

df_tc.to_csv("./ComputedData/youbike_Taichung.csv", index=False, encoding='utf-8')