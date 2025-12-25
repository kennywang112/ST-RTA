import pandas as pd

dataA1 = pd.DataFrame()
dataA1_113 = pd.read_csv('./Data/Accident_new/113年傷亡道路交通事故資料/113年度A1交通事故資料.csv').iloc[:-2 , :]
dataA1_114 = pd.read_csv('./Data/Accident_new/114年傷亡道路交通事故資料/A1_202508.csv').iloc[:-2 , :]
dataA1 = pd.concat([dataA1, dataA1_113, dataA1_114], ignore_index=True)
dataA1 = dataA1.astype({'發生年度': 'int32', '發生月份': 'int32'})

dataA1 = dataA1[~((dataA1['發生年度'] == 2025) & (dataA1['發生月份'] == 9))]
dataA1 = dataA1.astype({'發生年度': 'int32', '發生月份': 'int32'})

dataA2 = pd.DataFrame()
for i in range(1, 13):
    dataA2_month = pd.read_csv(f'./Data/Accident_new/113年傷亡道路交通事故資料/113年度A2交通事故資料_{i}.csv').iloc[:-2 , :]
    print(dataA2_month.shape)
    dataA2 = pd.concat([dataA2, dataA2_month], ignore_index=True)

# 依照A1資料量只取到今年8月
for i in range(1, 9):
    dataA2_month = pd.read_csv(f'./Data/Accident_new/114年傷亡道路交通事故資料/NPA_TMA2_0{i}/NPA_TMA2_{i}.csv').iloc[:-2 , :]
    print(dataA2_month.shape)
    dataA2 = pd.concat([dataA2, dataA2_month], ignore_index=True)

dataA2 = dataA2.astype({'發生年度': 'int32', '發生月份': 'int32'})

dataA1.to_csv('./Data/Accident/A1.csv', index=False)
dataA2.to_csv('./Data/Accident/A2.csv', index=False)