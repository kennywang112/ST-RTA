import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

res = requests.get('https://shaoweiwu088.pixnet.net/blog/post/262765884-%E5%85%A8%E5%8F%B0%E6%99%AF%E9%BB%9E%E5%BA%A7%E6%A8%99%E4%BD%8D%E7%BD%AE%E5%9C%96')
res.encoding = 'UTF-8'
soup = BeautifulSoup(res.text,'html.parser')
viewpoint = soup.find_all('tr', {'style':"height:16.5pt"})

def extract_viewpoint_name(text):
    lines = text.strip().splitlines()
    for line in lines:
        if line.strip():
            return line.strip()
    return None

data_list = []
for i in viewpoint:
    text = i.text

    name = extract_viewpoint_name(text)
    # 抓 DMS 格式（單個），像 25°08'37.0"N 或 121°47'58.6"E
    dms_single = re.findall(r'\d{2,3}°\d{1,2}\'\d{1,2}(\.\d+)?["]?[NSEW]?', text)

    # 抓合併的 DMS 經緯度，例如 '25°7\'59.56",121°40\'25.57"'
    dms_combined = re.findall(r'(\d{2,3}°\d{1,2}\'\d{1,2}(\.\d+)?"),\s*(\d{2,3}°\d{1,2}\'\d{1,2}(\.\d+)?")', text)

    # 抓十進位格式，像 25.133222
    decimal_coords = re.findall(r'\d{2,3}\.\d{6}', text)

    data_list.append({
        "raw": text,
        "attraction": name,
        "decimal_coords": decimal_coords
    })

df = pd.DataFrame(data_list)
print(df.shape)
# filter decimal_coords not empty
df = df[df['decimal_coords'].astype(bool)]
df[['attraction', 'decimal_coords']].to_csv('./ComputedData/Attractions.csv', index=False, encoding='utf-8')