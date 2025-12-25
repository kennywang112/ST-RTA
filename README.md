# ST-RTA: Spatio-Temporal Road Traffic Accident
Reanalyze it starting from [here](https://github.com/kennywang112/TrafficTDApython), but apply spatial and temporal analysis techniques.
**MacroCounty** and **MicroCounty** calculates nearest neighbor based on county, while **MacroGrid** and **MicroGrid** is based on Grid we generated.
**Models** folder include not only Mapper Algorithm from TDA, but also three machine learning methods to identify hotspot we generated from Getis-ord.

### Data Sources:
1. [Youbike (deprecated)](https://data.gov.tw/suggests/136458)
([New Taipei](https://data.gov.tw/dataset/146969),
[Taipei](https://data.gov.tw/dataset/137993),
[Taichung](https://data.gov.tw/dataset/136781))
2. [Parking Lot](https://data.gov.tw/suggests/136651?utm_source=chatgpt.com)
3. [Metro Rapid Transit](https://tdx.transportdata.tw/api-service/swagger/basic/945f57da-f29d-4dfd-94ec-c35d9f62be7d#/): Transport Data Exchange(捷運站出入口基本資料)
4. [Youbike](https://tdx.transportdata.tw/api-service/swagger/basic/945f57da-f29d-4dfd-94ec-c35d9f62be7d#/): 自行車租借站位資料
5. [Taiwan shape](https://data.gov.tw/dataset/7442) and [鄉鎮市區界線(TWD97經緯度)1140318](https://whgis-nlsc.moi.gov.tw/Opendata/Files.aspx)

### Additional Attributes
1. Attractions
2. School Data

### Data source Update
- Youbike, MRT, Attractions data sources installed in 2025/04/15<br/>
- Parkinglot data sources installed in 2025/04/21<br/>
- Taiwan shape 2025/06/06
- Traffic accident data installed in 2025/10/18, include data until Sep

### Env
```shell
conda create --name ST-RTA python=3.10
conda activate ST-RTA
pip install -r requirements.txt
```

### Core
- DataCombine: Combine `A1` and `A2` separately in each year, month<br/>
- DataPreprocess: Preprocess data for grid analyze `grid_gi` and model input data `all_features`<br/>
- Model: Run ML model from `all_features`<br/>
- FilterforMapper: Get the filter function for Mapper from `all_features` data<br/>
- GetCCforCombineddata: Add County name to combined data, get `combineddata_with_CC`

### Paper
Some reference papers are stored in [notion](https://www.notion.so/Spatio-temporal-Analysis-Paper-1f275012ce1a800086e2cf4a2b1b3075?source=copy_link)