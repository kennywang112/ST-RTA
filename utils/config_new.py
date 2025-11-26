select_group = [
    # 氣候暫不討論
    # '天候名稱', '光線名稱',

    # 道路問題
    '路面狀況-路面鋪裝名稱', '路面狀況-路面狀態名稱', '路面狀況-路面缺陷名稱',
    '道路障礙-障礙物名稱', '道路障礙-視距品質名稱', '道路障礙-視距名稱',

    # 號誌
    '號誌-號誌種類名稱',# '號誌-號誌動作名稱',

    # 車道劃分
    '車道劃分設施-分道設施-快車道或一般車道間名稱',
    '車道劃分設施-分道設施-快慢車道間名稱', '車道劃分設施-分道設施-路面邊線名稱',

    # 大類別
    # '肇因研判大類別名稱-主要', '肇因研判大類別名稱-個別', # 聚焦道路類型
    '當事者區分-類別-大類別名稱-車種', # 聚焦道路類型
    # '當事者行動狀態大類別名稱', # 聚焦道路類型
    '車輛撞擊部位大類別名稱-最初', #'車輛撞擊部位大類別名稱-其他',
    '事故類型及型態大類別名稱', '車道劃分設施-分向設施大類別名稱',
    # '事故位置大類別名稱', # 和道路型態大類別名稱相同
    '道路型態大類別名稱',
    
    # 子類別
    # '肇因研判子類別名稱-主要', '肇因研判子類別名稱-個別', # 聚焦道路類型
    'cause_group',
    # '當事者區分-類別-子類別名稱-車種', # 聚焦道路類型
    # '當事者行動狀態子類別名稱', # 聚焦道路類型
    # '車輛撞擊部位子類別名稱-最初', '車輛撞擊部位子類別名稱-其他', # 道路類型很大程度影響撞擊部位，所以不考慮
    # '事故類型及型態子類別名稱', '車道劃分設施-分向設施子類別名稱', 
    # '事故位置子類別名稱', '道路型態子類別名稱',

    # 其他
    # '當事者屬-性-別名稱', '當事者事故發生時年齡', 
    '速限-第1當事者', '道路類別-第1當事者-名稱',
    # '保護裝備名稱', '行動電話或電腦或其他相類功能裝置名稱', '肇事逃逸類別名稱-是否肇逃',

    # 設施
    'youbike_100m_count', 'mrt_100m_count', 'parkinglot_100m_count',

    # A1 or A2
    # 'source',
    ]

for_poly = [
    '號誌-號誌種類名稱',
    '車道劃分設施-分道設施-快車道或一般車道間名稱',
    '車道劃分設施-分道設施-快慢車道間名稱',
    '車道劃分設施-分道設施-路面邊線名稱',
    '車輛撞擊部位大類別名稱-最初',
    '事故類型及型態大類別名稱',
    '車道劃分設施-分向設施大類別名稱',
    '當事者區分-類別-大類別名稱-車種',
    '道路型態大類別名稱',
    '速限-第1當事者',
    '道路類別-第1當事者-名稱',
    'youbike_100m_count',
    'mrt_100m_count',
    'parkinglot_100m_count',
    'county',
    'cause-group'
 ]

group_translation = {
    "道路類別-第1當事者-名稱": "Road category – Party 1",
    "速限-第1當事者": "Speed limit – Party 1",
    "號誌-號誌種類名稱": "Traffic signal type",
    "車道劃分設施-分向設施大類別名稱": "Lane division facility – directional separation (major category)",
    "車輛撞擊部位大類別名稱-最初": "Vehicle impact location (major category, initial)",
    "county": "County",
    "車道劃分設施-分道設施-快車道或一般車道間名稱": "Lane Division Facility - Between Fast and General Lanes",
    "車道劃分設施-分道設施-快慢車道間名稱": "Lane Division Facility - Between Fast and Slow Lanes",
    "車道劃分設施-分道設施-路面邊線名稱": "Lane Division Facility - Road Edge Line",
    "道路型態大類別名稱": "Road type (major category)",
    "事故類型及型態大類別名稱": "Accident type and pattern (major category)",
    "youbike": "YouBike",
    "parkinglot": "Parking lot",
    "道路障礙-視距名稱": "Road obstacle – Sight distance",
    "路面狀況-路面狀態名稱": "Road surface condition – Surface status",
    "mrt": "MRT",
    "路面狀況-路面鋪裝名稱": "Road surface condition – Pavement type",
    "道路障礙-視距品質名稱": "Road obstacle – Sight distance quality",
    "道路障礙-障礙物名稱": "Road obstacle – Obstacle type",
    "路面狀況-路面缺陷名稱": "Road surface condition – Defect type",
    "original-speed": "Original Speed",
    "當事者區分-類別-大類別名稱-車種": "Party classification – Vehicle type (major category)",
    "cause-group": "Cause",
}