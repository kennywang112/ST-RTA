select_group = [
    # 氣候暫不討論
    # '天候名稱', '光線名稱',

    # 道路問題
    '路面狀況-路面鋪裝名稱', '路面狀況-路面狀態名稱', '路面狀況-路面缺陷名稱',
    '道路障礙-障礙物名稱', '道路障礙-視距品質名稱', '道路障礙-視距名稱',

    # 號誌
    '號誌-號誌種類名稱', '號誌-號誌動作名稱',

    # 車道劃分
    '車道劃分設施-分道設施-快車道或一般車道間名稱',
    '車道劃分設施-分道設施-快慢車道間名稱', '車道劃分設施-分道設施-路面邊線名稱',

    # 大類別
    # '肇因研判大類別名稱-主要', '肇因研判大類別名稱-個別', # 聚焦道路類型
    # '當事者區分-類別-大類別名稱-車種', # 聚焦道路類型
    # '當事者行動狀態大類別名稱', # 聚焦道路類型
    '車輛撞擊部位大類別名稱-最初', #'車輛撞擊部位大類別名稱-其他',
    '事故類型及型態大類別名稱', '車道劃分設施-分向設施大類別名稱',
    # '事故位置大類別名稱', # 和道路型態大類別名稱相同
    '道路型態大類別名稱',
    
    # 子類別
    # '肇因研判子類別名稱-主要', '肇因研判子類別名稱-個別', # 聚焦道路類型
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

select_group_behaviour = [
    # 號誌
    '號誌-號誌種類名稱', '號誌-號誌動作名稱',

    # 車道劃分
    '車道劃分設施-分道設施-快車道或一般車道間名稱',
    '車道劃分設施-分道設施-快慢車道間名稱', '車道劃分設施-分道設施-路面邊線名稱',

    # 大類別
    '事故類型及型態子類別名稱', '車道劃分設施-分向設施子類別名稱',
    '道路型態子類別名稱',

    # 其他
    '速限-第1當事者',
    '道路類別-第1當事者-名稱',

    # 設施
    'youbike_100m_count',
    
    # 駕駛、行人行為
    '肇因研判子類別名稱-主要',

    'COUNTYNAME'
    ]

# 在model preprocess被使用，用意是將這幾個欄位作為交互作用的欄位，不使用select_group的原因在於路面狀況等資料太多且和道路設計較無關
for_poly = [
    '號誌-號誌種類名稱',
    # '號誌-號誌動作名稱',
    '車道劃分設施-分道設施-快車道或一般車道間名稱',
    '車道劃分設施-分道設施-快慢車道間名稱',
    '車道劃分設施-分道設施-路面邊線名稱',
    '車輛撞擊部位大類別名稱-最初',
    '事故類型及型態大類別名稱',
    '車道劃分設施-分向設施大類別名稱',
    # '事故位置大類別名稱', 
    '道路型態大類別名稱',
    '速限-第1當事者',
    '道路類別-第1當事者-名稱',
    'youbike_100m_count',
    'mrt_100m_count',
    'parkinglot_100m_count',
    'county'
 ]

# For english version

feature_name_map = {
    '號誌-號誌種類名稱': 'Traffic signal type',

    '車道劃分設施-分道設施-快車道或一般車道間名稱': 'Lane division facility - separation between expressway and general lane',
    '車道劃分設施-分道設施-快慢車道間名稱': 'Lane division facility - separation between fast and slow lanes',
    '車道劃分設施-分道設施-路面邊線名稱': 'Lane division facility - edge line marking',

    '事故類型及型態子類別名稱': 'Accident type and pattern',
    '車道劃分設施-分向設施子類別名稱': 'Lane division facility - directional separation',
    '道路型態子類別名稱': 'Road type',

    '速限-第1當事者': 'Speed limit',
    '道路類別-第1當事者-名稱': 'Road category',

    '肇因研判子類別名稱-主要': 'Primary cause determination'
}

category_value_map = {
    '號誌-號誌種類名稱': {
        '無號誌': 'No signal',
        '閃光號誌': 'Flashing signal',
        '行車管制號誌(附設行人專用號誌)': 'Traffic control signal (with pedestrian-only signal)',
        '行車管制號誌': 'Traffic control signal'
    },

    '車道劃分設施-分道設施-快車道或一般車道間名稱': {
        '車道線(附標記)': 'Lane line (with marking)',
        '未繪設車道線': 'No lane line',
        '車道線(無標記)': 'Lane line (without marking)',
        '禁止變換車道線(附標記)': 'No-lane-change line (with marking)',
        '禁止變換車道線(無標記)': 'No-lane-change line (without marking)'
    },

    '車道劃分設施-分道設施-快慢車道間名稱': {
        '車道線(附標記)': 'Lane line (with marking)',
        '未繪設車道線': 'No lane line',
        '車道線(無標記)': 'Lane line (without marking)',
        '禁止變換車道線(附標記)': 'No-lane-change line (with marking)',
        '禁止變換車道線(無標記)': 'No-lane-change line (without marking)'
    },

    '車道劃分設施-分道設施-路面邊線名稱': {
        '有': 'Present',
        '無': 'Absent'
    },

    '事故類型及型態子類別名稱': {
        '撞路樹': 'Hit tree',
        '路口交岔撞': 'Intersection collision',
        '撞建築物': 'Hit building',
        '衝出路外': 'Run off road',
        '追撞': 'Rear-end collision',
        '路上翻車、摔倒': 'Rollover/Fall on road',
        '其他': 'Other',
        '撞電桿': 'Hit utility pole',
        '側撞': 'Side collision',
        '穿越道路中': 'Crossing road',
        '同向擦撞': 'Same-direction sideswipe',
        '對撞': 'Head-on collision',
        '撞交通島': 'Hit traffic island',
        '撞護欄(樁)': 'Hit guardrail (post)',
        '同向通行中': 'Traveling in same direction',
        '撞橋樑(橋墩)': 'Hit bridge (pier)',
        '撞工程施工': 'Hit construction site',
        '對向通行中': 'Opposite-direction traveling',
        '撞號誌、標誌桿': 'Hit signal/sign pole',
        '倒車撞': 'Backing collision',
        '對向擦撞': 'Opposite-direction sideswipe',
        '撞動物': 'Hit animal',
        '佇立路邊(外)': 'Standing roadside (outside)',
        '在路上作業中': 'Working on road',
        '衝進路中': 'Rush into roadway',
        '撞非固定設施': 'Hit non-fixed facility',
        '衝過(或撞壞)遮斷器': 'Break through (or hit) barrier gate',
        '從停車後(或中)穿出': 'Emerge from parking',
        '在路上嬉戲': 'Playing on road',
        '正越過平交道中': 'Crossing level crossing'
    },

    '車道劃分設施-分向設施子類別名稱': {
        '窄式無柵欄': 'Narrow type without barrier',
        '寬式(50公分以上)': 'Wide type (over 50 cm)',
        '附標記': 'With marking',
        '無分向設施': 'No directional separation facility',
        '無標記': 'Without marking',
        '窄式附柵欄': 'Narrow type with barrier'
    },

    '道路型態子類別名稱': {
        '直路': 'Straight road',
        '三岔路': 'T-junction',
        '四岔路': 'Crossroad',
        '彎曲路及附近': 'Curved road and vicinity',
        '涵洞': 'Culvert',
        '多岔路': 'Multi-junction',
        '坡路': 'Slope road',
        '橋樑': 'Bridge',
        '地下道': 'Underpass',
        '隧道': 'Tunnel',
        '其他': 'Other',
        '高架道路': 'Elevated road',
        '圓環': 'Roundabout',
        '有遮斷器': 'With barrier gate',
        '休息站或服務區': 'Rest area or service area',
        '廣場': 'Plaza'
    },

    '道路類別-第1當事者-名稱': {
        '省道': 'Provincial highway',
        '市區道路': 'Urban road',
        '縣道': 'County road',
        '村里道路': 'Village road',
        '國道': 'National highway',
        '其他': 'Other',
        '快速(公)道': 'Expressway',
        '鄉道': 'Township road',
        '專用道路': 'Exclusive road'
    },

    '肇因研判子類別名稱-主要': {
        '患病或服用藥物(疲勞)駕駛': 'Fatigue driving due to illness or medication',
        '起步時未注意安全': 'Failure to ensure safety when starting',
        '酒醉(後)駕駛': 'Drunk driving',
        '違反閃光號誌': 'Violation of flashing signal',
        '恍神、緊張、心不在焉分心駕駛': 'Distracted driving due to drowsiness,\n nervousness, or absent-mindedness',
        '未保持行車安全距離': 'Failure to maintain safe driving distance',
        '車輛或機械操作不當(慎)': 'Improper vehicle/mechanical operation (caution)',
        '車輛未依規定暫停讓行人先行': 'Failure to yield to pedestrians as required',
        '其他不當駕車行為': 'Other improper driving behaviors',
        '闖紅燈左轉(或迴轉)': 'Running red light when turning left (or U-turn)',
        '闖紅燈直行': 'Running red light when going straight',
        '觀看其他事故、活動、道路環境或車外資訊分心駕駛': 'Distracted driving due to watching accidents, events,\n road environment, or external information',
        '無號誌路口，轉彎車未讓直行車先行': 'Unsignalized intersection -\n turning vehicle failed to yield to through vehicle',
        '在道路上嬉戲或奔走不定': 'Playing or running on the road',
        '左轉彎未依規定': 'Improper left turn',
        '逆向行駛': 'Wrong-way driving',
        '尚未發現肇事因素': 'Cause of accident not yet identified',
        '無號誌路口，左方車未讓右方車先行': 'Unsignalized intersection -\n vehicle from the left failed to yield to vehicle from the right',
        '右轉彎未依規定': 'Improper right turn',
        '未依標誌或標線穿越道路': 'Failure to cross road according to signs or markings',
        '未依規定減速': 'Failure to slow down as required',
        '相關跡證不足且無具體影像紀錄，當事人各執一詞，經分析後無法釐清肇事原因': 'Insufficient evidence and no concrete video record,\n conflicting statements from parties, unable to clarify cause',
        '閃避不當(慎)': 'Improper evasive maneuver (caution)',
        '有號誌路口，轉彎車未讓直行車先行': 'Signalized intersection -\n turning vehicle failed to yield to through vehicle',
        '違反禁止超車標誌(線)': 'Violation of no-overtaking sign/line',
        '打瞌睡或疲勞駕駛(包括連續駕車8小時)': 'Drowsy or fatigued driving\n (including continuous driving over 8 hours)',
        '變換車道不當': 'Improper lane change',
        '倒車未依規定': 'Improper reversing',
        '違反二段式左(右)轉標誌(線)': 'Violation of two-stage left/right turn sign/line',
        '無號誌路口，支線道未讓幹線道先行': 'Unsignalized intersection -\n minor road failed to yield to major road',
        '其他未依規定讓車': 'Other failure to yield',
        '未保持行車安全間隔': 'Failure to maintain safe driving interval',
        '方向不定(不包括危險駕車)': 'Unsteady driving direction (excluding dangerous driving)',
        '違反其他標誌(線)禁制': 'Violation of other restrictive signs/lines',
        '迴轉未依規定': 'Improper U-turn',
        '因光線、視線遮蔽致生事故': 'Accident due to light/visibility obstruction',
        '未依號誌或手勢指揮(示)穿越道路': 'Failure to cross road according\n to signals or hand gestures',
        '超速駕駛': 'Speeding',
        '停車操作時未注意安全': 'Failure to ensure safety when parking',
        '違反其他號誌': 'Violation of other signals',
        '飲食、抽(點)菸、拿(撿)物品分心駕駛': 'Distracted driving due to eating, smoking, or picking objects',
        '違規(臨時)停車': 'Illegal (temporary) parking',
        '吸食違禁物駕駛': 'Driving under influence of illegal drugs',
        '其他引起事故之疏失或行為': 'Other errors or behaviors causing accidents',
        '無號誌路口，少線道未讓多線道先行': 'Unsignalized intersection -\n fewer lanes failed to yield to more lanes',
        '開啟或關閉車門不當': 'Improper door opening/closing',
        '穿越道路未注意左右來車': 'Crossing road without checking traffic',
        '橫越道路不慎': 'Careless road crossing',
        '違規超車': 'Illegal overtaking',
        '在道路上工作之人員未設適當標識': 'Road workers without proper warning signs',
        '闖紅燈右轉': 'Running red light when turning right',
        '車輪脫落或輪胎爆裂': 'Wheel detachment or tire blowout',
        '違反禁止左轉、右轉標誌': 'Violation of no left/right turn sign',
        '車輛拋錨未採安全措施': 'Vehicle breakdown without safety measures',
        '未避讓(跟隨、併駛、超車)消防、救護、警備、工程救險車、毒性化學物質災害事故應變車等執行緊急任務車': 'Failure to yield to emergency vehicles \n(fire, ambulance, police, engineering rescue, hazardous material response)',
        '未靠右行駛': 'Failure to keep right',
        '違反禁止變換車道標線': 'Violation of no lane-change marking',
        '爭(搶)道行駛': 'Aggressive driving (road hogging)',
        '未依規定行走地下道、天橋穿越道路': 'Failure to use underpass/overpass\n as required when crossing',
        '車輛零件脫落': 'Vehicle parts detachment',
        '違反禁行車種標誌(字)': 'Violation of prohibited vehicle type sign',
        '多車道迴轉，未先駛入內側車道': 'Multi-lane U-turn without\n first moving into inner lane',
        '載運貨物超長、寬、高': 'Overlength/width/height cargo load',
        '搶(闖)越平交道': 'Running through level crossing',
        '危險駕駛': 'Dangerous driving',
        '違反車輛專用標誌(線)': 'Violation of vehicle-only sign/line',
        '違反遵行方向標誌(線)': 'Violation of mandatory direction sign/line',
        '道路設施(備)、植栽或其他裝置，倒塌或掉(斷)落': 'Collapse or falling of road facility,\n vegetation, or other installation',
        '其他機件失靈或故障': 'Other mechanical failure or malfunction',
        '動物竄出': 'Animal darting into road',
        '未依規定使用燈光': 'Failure to use lights properly',
        '煞車失靈或故障': 'Brake failure or malfunction',
        '其他交通管制不當': 'Other improper traffic control',
        '裝載貨物不穩妥': 'Unstable cargo loading',
        '違反禁止迴轉或迴車標誌': 'Violation of no U-turn/turnaround sign',
        '乘客、車上動(生)物干擾分心駕駛': 'Distracted driving due to\n passengers or animals inside vehicle',
        '肇事逃逸未查獲，無法查明肇因': 'Hit-and-run (unresolved cause)',
        '行經圓環未依規定讓車': 'Failure to yield at roundabout',
        '操作、觀看行車輔助或娛樂性顯示設備': 'Operating or watching\n driver-assist/entertainment device',
        '其他裝載不當': 'Other improper cargo loading',
        '未遵守依法令授權交通指揮人員之指揮': 'Failure to obey authorized\n traffic officer’s command',
        '車輛未停妥滑動致生事故': 'Vehicle sliding accident due to not properly parked',
        '山路會車，靠山壁車未讓外緣車先行': 'Mountain road passing -\n vehicle near mountain wall failed to yield to outer edge vehicle',
        '裝載未盡安全措施': 'Failure to secure cargo loading',
        '暗處停車無燈光、標識': 'Parking in dark area without lights/signs',
        '違反行人專用標誌(線)': 'Violation of pedestrian-only sign/line',
        '物品(件)滾(滑行)或飛(掉)落': 'Objects rolling/sliding/falling off',
        '未待乘客安全上下而開車': 'Driving off before passengers safely boarded/alighted',
        '被車輛輾壓之不明物體彈飛': 'Collision with unidentified object on road',
        '使用手持行動電話': 'Using handheld mobile phone',
        '燈光系統故障': 'Lighting system failure',
        '施工安全防護措施未依規定或未盡完善(備)': 'Construction safety measures\n not compliant or inadequate',
        '違反禁止進入標誌': 'Violation of no-entry sign',
        '違反禁止各種車輛進入標誌': 'Violation of no vehicles allowed sign',
        '車輛附屬機具或車門未盡安全措施': 'Failure to secure auxiliary\n equipment or vehicle doors',
        '事故發生時當事者逕自離開現場': 'Leaving accident scene without consent',
        '超載人員': 'Overloading passengers',
        '夜間行駛無燈光設備': 'Night driving without lights',
        '載運貨物超重': 'Overloaded cargo',
        '平交道看守疏失或未放柵欄': 'Level crossing guard error or barrier not lowered',
        '峻狹坡路會車，下坡車未讓上坡車先行': 'Steep narrow slope - downhill\n vehicle failed to yield to uphill vehicle',
        '使用車輛自動駕駛或先進駕駛輔助系統設備(裝置)不符規定': 'Improper use of autonomous or\n advanced driver-assist system',
        '方向操縱系統故障': 'Steering system failure',
        '路況危險無安全(警告)設施': 'Dangerous road conditions\n without safety/warning facilities',
        '裝卸貨物不當': 'Improper cargo handling',
        '乘坐不當(慎)': 'Improper passenger seating (caution)',
        '強風、暴雨、濃霧(煙)': 'Strong wind, heavy rain, dense fog (smoke)',
        '發生事故後，未採取安全措施': 'Failure to take safety measures after accident',
        '違反車輛改道標誌': 'Violation of detour sign',
        '上下車輛時未注意安全': 'Failure to ensure safety when boarding/alighting',
        '未待車輛停妥而上下車': 'Boarding/alighting before vehicle fully stopped',
        '違反禁止會車標誌': 'Violation of no meeting sign',
        '未依法令授權指揮交通或指揮不當': 'Failure to follow authorized\n traffic direction or improper command'
    },

    '速限-第1當事者': {
        '': ''
    }
}

countycity_dct = {
    '臺南市': 'Tainan City',
    '高雄市': 'Kaohsiung City',
    '嘉義市': 'Chiayi City',
    '嘉義縣': 'Chiayi County',
    '屏東縣': 'Pingtung County',
    '彰化縣': 'Changhua County',
    '雲林縣': 'Yunlin County',
    '臺中市': 'Taichung City',
    '苗栗縣': 'Miaoli County',
    '新竹市': 'Hsinchu City',
    '新竹縣': 'Hsinchu County',
    '臺東縣': 'Taitung County',
    '桃園市': 'Taoyuan City',
    '新北市': 'New Taipei City',
    '臺北市': 'Taipei City',
    '花蓮縣': 'Hualien County',
    '宜蘭縣': 'Yilan County',
    '基隆市': 'Keelung City',
    '南投縣': 'Nantou County'
}

# used in Model.ipynb
col_translation = {
    "parkinglot_100m_count_mean": "Parking lot count within 100m (mean)",
    "速限-第1當事者_mean": "Speed limit - Party 1 (mean)",
    "車道劃分設施-分向設施大類別名稱_無": "Lane division facility - directional separation (major category)_None",
    "車道劃分設施-分道設施-快車道或一般車道間名稱_車道線(無標記)": "Lane Division Facility - Between Fast and General Lanes_Lane line (without marking)",
    "道路型態大類別名稱_交岔路": "Road type (major category)_Intersection",
    "道路類別-第1當事者-名稱_市區道路": "Road category - Party 1_Urban road",
    "道路類別-第1當事者-名稱_省道": "Road category - Party 1_Provincial highway",
    "車道劃分設施-分道設施-路面邊線名稱_有": "Lane Division Facility - Road Edge Line_Present",
    "車道劃分設施-分道設施-快車道或一般車道間名稱_未繪設車道線": "Lane Division Facility - Between Fast and General Lanes_No lane line",
    "車道劃分設施-分道設施-快慢車道間名稱_快慢車道分隔線": "Lane Division Facility - Between Fast and Slow Lanes_Dividing line",
    "號誌-號誌種類名稱_閃光號誌": "Traffic signal type_Flashing signal",
    "車道劃分設施-分向設施大類別名稱_單向禁止超車線": "Lane division facility - directional separation (major category)_One-way no-overtaking line",
    "事故類型及型態大類別名稱_車輛本身": "Accident type and pattern (major category)_Vehicle itself",
    "道路類別-第1當事者-名稱_村里道路": "Road category - Party 1_Village road",
    "車道劃分設施-分道設施-快慢車道間名稱_窄式快慢車道分隔島(附柵欄)": "Lane Division Facility - Between Fast and Slow Lanes_Narrow-type divider island (with barrier)",
    "車道劃分設施-分道設施-快慢車道間名稱_窄式快慢車道分隔島(無柵欄)": "Lane Division Facility - Between Fast and Slow Lanes_Narrow-type divider island (without barrier)",
    "車道劃分設施-分道設施-快車道或一般車道間名稱_車道線(附標記)": "Lane Division Facility - Between Fast and General Lanes_Lane line (with marking)",
    "道路類別-第1當事者-名稱_國道": "Road category - Party 1_National highway",
    "道路類別-第1當事者-名稱_專用道路": "Road category - Party 1_Exclusive road",
    "道路類別-第1當事者-名稱_鄉道": "Road category - Party 1_Township road",
    "道路型態大類別名稱_單路部分": "Road type (major category)_Single road section",
    "事故類型及型態大類別名稱_車與車": "Accident type and pattern (major category)_Vehicle-to-vehicle",
    "車道劃分設施-分向設施大類別名稱_行車分向線": "Lane division facility - directional separation (major category)_Directional line",
    "號誌-號誌種類名稱_無號誌": "Traffic signal type_No signal",
    "車輛撞擊部位大類別名稱-最初_其他": "Vehicle impact location (major category, initial)_Other",
    "車輛撞擊部位大類別名稱-最初_機車與自行車": "Vehicle impact location (major category, initial)_Motorcycle/Bicycle",
    "號誌-號誌種類名稱_行車管制號誌": "Traffic signal type_Traffic control signal",
    "車道劃分設施-分道設施-快車道或一般車道間名稱_禁止變換車道線(無標記)": "Lane Division Facility - Between Fast and General Lanes_No-lane-change line (without marking)",
    "車道劃分設施-分向設施大類別名稱_中央分向島": "Lane division facility - directional separation (major category)_Central divider island",
    "事故類型及型態大類別名稱_人與車": "Accident type and pattern (major category)_Person vs vehicle",
    "車輛撞擊部位大類別名稱-最初_汽車": "Vehicle impact location (major category, initial)_Automobile",
    "道路類別-第1當事者-名稱_其他": "Road category - Party 1_Other",
    "道路型態大類別名稱_其他": "Road type (major category)_Other",
    "號誌-號誌種類名稱_行車管制號誌(附設行人專用號誌)": "Traffic signal type_Traffic control signal (with pedestrian-only signal)",
    "車道劃分設施-分道設施-快車道或一般車道間名稱_禁止變換車道線(附標記)": "Lane Division Facility - Between Fast and General Lanes_No-lane-change line (with marking)",
    "車道劃分設施-分道設施-快慢車道間名稱_未繪設快慢車道分隔線": "Lane Division Facility - Between Fast and Slow Lanes_No lane line",
    "車道劃分設施-分道設施-路面邊線名稱_無": "Lane Division Facility - Road Edge Line_Not present",
    "車道劃分設施-分道設施-快慢車道間名稱_寬式快慢車道分隔島(50公分以上)": "Lane Division Facility - Between Fast and Slow Lanes_Wide-type divider island (≥50 cm)",
    "車道劃分設施-分向設施大類別名稱_雙向禁止超車線": "Lane division facility - directional separation (major category)_Two-way no-overtaking line",
    "道路類別-第1當事者-名稱_縣道": "Road category - Party 1_County road",
    "事故類型及型態大類別名稱_平交道事故": "Accident type and pattern (major category)_Level crossing accident",
    "道路型態大類別名稱_平交道": "Road type (major category)_Level crossing",
    "youbike_100m_count_mean": "YouBike station count within 100m (mean)",
    "道路類別-第1當事者-名稱_快速(公)道": "Road category - Party 1_Expressway",
    "道路型態大類別名稱_圓環廣場": "Road type (major category)_Roundabout/Plaza",
    "mrt_100m_count_mean": "MRT station count within 100m (mean)",
    "當事者區分-類別-大類別名稱-車種_大客車": "Party category - Vehicle type_Large bus",
    "當事者區分-類別-大類別名稱-車種_小貨車": "Party category - Vehicle type_Small truck",
    "當事者區分-類別-大類別名稱-車種_全聯結車": "Party category - Vehicle type_Full trailer truck",
    "當事者區分-類別-大類別名稱-車種_其他車": "Party category - Vehicle type_Other vehicle",
    "當事者區分-類別-大類別名稱-車種_軍車": "Party category - Vehicle type_Military vehicle",
    "當事者區分-類別-大類別名稱-車種_特種車": "Party category - Vehicle type_Special-purpose vehicle",
    "當事者區分-類別-大類別名稱-車種_小貨車(含客、貨兩用)": "Party category - Vehicle type_Small truck (incl. passenger/cargo)",
    "當事者區分-類別-大類別名稱-車種_小客車(含客、貨兩用)": "Party category - Vehicle type_Passenger car (incl. passenger/cargo)",
    "當事者區分-類別-大類別名稱-車種_半聯結車": "Party category - Vehicle type_Semi-trailer truck",
    "當事者區分-類別-大類別名稱-車種_機車": "Party category - Vehicle type_Motorcycle",
    "當事者區分-類別-大類別名稱-車種_曳引車": "Party category - Vehicle type_Tractor unit",
    "當事者區分-類別-大類別名稱-車種_慢車": "Party category - Vehicle type_Slow-moving vehicle",
    "當事者區分-類別-大類別名稱-車種_人": "Party category - Vehicle type_Pedestrian",
    "當事者區分-類別-大類別名稱-車種_大貨車": "Party category - Vehicle type_Heavy truck",

    #  (Pavement)
    '路面狀況-路面鋪裝名稱_柏油': 'Road Surface - Pavement_Asphalt',
    '路面狀況-路面鋪裝名稱_水泥': 'Road Surface - Pavement_Concrete',
    '路面狀況-路面鋪裝名稱_碎石': 'Road Surface - Pavement_Gravel',
    '路面狀況-路面鋪裝名稱_無鋪裝': 'Road Surface - Pavement_Unpaved',
    '路面狀況-路面鋪裝名稱_其他鋪裝': 'Road Surface - Pavement_Other',

    # (Condition)
    '路面狀況-路面狀態名稱_乾燥': 'Road Surface - Condition_Dry',
    '路面狀況-路面狀態名稱_濕潤': 'Road Surface - Condition_Wet',
    '路面狀況-路面狀態名稱_泥濘': 'Road Surface - Condition_Muddy',
    '路面狀況-路面狀態名稱_油滑': 'Road Surface - Condition_Slippery',
    '路面狀況-路面狀態名稱_冰雪': 'Road Surface - Condition_Icy/Snowy',

    # (Defect)
    '路面狀況-路面缺陷名稱_無缺陷': 'Road Surface - Defect_None',
    '路面狀況-路面缺陷名稱_路面鬆軟': 'Road Surface - Defect_Soft',
    '路面狀況-路面缺陷名稱_有坑洞': 'Road Surface - Defect_Pothole',
    '路面狀況-路面缺陷名稱_突出(高低)不平': 'Road Surface - Defect_Uneven',

    # (Obstacle Type)
    '道路障礙-障礙物名稱_無障礙物': 'Road Obstacle - Type_None',
    '道路障礙-障礙物名稱_路上有停車': 'Road Obstacle - Type_Parked Vehicle',
    '道路障礙-障礙物名稱_道路工事(程)中': 'Road Obstacle - Type_Under Construction',
    '道路障礙-障礙物名稱_攤位、棚架': 'Road Obstacle - Type_Stall/Scaffold',
    '道路障礙-障礙物名稱_有堆積物': 'Road Obstacle - Type_Debris',
    '道路障礙-障礙物名稱_其他障礙物': 'Road Obstacle - Type_Other',

    # (Sight Distance Quality)
    '道路障礙-視距品質名稱_無遮蔽物': 'Sight Distance - Quality_Unobstructed',
    '道路障礙-視距品質名稱_有遮蔽物': 'Sight Distance - Quality_Obstructed',

    # (Sight Distance Obstacle)
    '道路障礙-視距名稱_良好': 'Sight Distance - Obstacle_Good',
    '道路障礙-視距名稱_彎道': 'Sight Distance - Obstacle_Curve',
    '道路障礙-視距名稱_坡道': 'Sight Distance - Obstacle_Slope',
    '道路障礙-視距名稱_建築物': 'Sight Distance - Obstacle_Building',
    '道路障礙-視距名稱_樹木、農作物': 'Sight Distance - Obstacle_Trees/Crops',
    '道路障礙-視距名稱_路上停放車輛': 'Sight Distance - Obstacle_Parked Vehicle',

    # (Traffic Signal Action)
    '號誌-號誌動作名稱_正常': 'Traffic Signal - Action_Normal',
    '號誌-號誌動作名稱_不正常': 'Traffic Signal - Action_Abnormal',
    '號誌-號誌動作名稱_無動作': 'Traffic Signal - Action_No Action',
    '號誌-號誌動作名稱_無號誌': 'Traffic Signal - Action_No Signal'

}

cause_mapping = {
    # 不專心
    "Distraction": [
        "起步時未注意安全",
        "恍神、緊張、心不在焉分心駕駛",
        "觀看其他事故、活動、道路環境或車外資訊分心駕駛",
        "飲食、抽(點)菸、拿(撿)物品分心駕駛",
        "穿越道路未注意左右來車",
        "橫越道路不慎",
        "乘客、車上動(生)物干擾分心駕駛",
        "操作、觀看行車輔助或娛樂性顯示設備",
        "使用手持行動電話",
    ],

    # 錯誤決策
    "Decision": [
        "違反閃光號誌",
        "車輛未依規定暫停讓行人先行",
        "其他不當駕車行為",
        "闖紅燈左轉(或迴轉)",
        "闖紅燈直行",
        "無號誌路口，轉彎車未讓直行車先行",
        "左轉彎未依規定",
        "逆向行駛",
        "無號誌路口，左方車未讓右方車先行",
        "右轉彎未依規定",
        "未依標誌或標線穿越道路",
        "未依規定減速",
        "閃避不當(慎)",
        "有號誌路口，轉彎車未讓直行車先行",
        "違反禁止超車標誌(線)",
        "變換車道不當",
        "倒車未依規定",
        "違反二段式左(右)轉標誌(線)",
        "無號誌路口，支線道未讓幹線道先行",
        "其他未依規定讓車",
        "未保持行車安全間隔",
        "未保持行車安全距離",
        "違反其他標誌(線)禁制",
        "迴轉未依規定",
        "未依號誌或手勢指揮(示)穿越道路",
        "超速駕駛",
        "違反其他號誌",
        "無號誌路口，少線道未讓多線道先行",
        "違規超車",
        "闖紅燈右轉",
        "違反禁止左轉、右轉標誌",
        "未避讓緊急任務車輛",
        "未避讓(跟隨、併駛、超車)消防、救護、警備、工程救險車、毒性化學物質災害事故應變車等執行緊急任務車",
        "未靠右行駛",
        "違反禁止變換車道標線",
        "爭(搶)道行駛",
        "未依規定行走地下道、天橋穿越道路",
        "多車道迴轉，未先駛入內側車道",
        "違反車輛專用標誌(線)",
        "違反遵行方向標誌(線)",
        "未依規定使用燈光",
        "違反禁止迴轉或迴車標誌",
        "行經圓環未依規定讓車",
        "未遵守依法令授權交通指揮人員之指揮",
        "山路會車，靠山壁車未讓外緣車先行",
        "違反行人專用標誌(線)",
        "違反車輛改道標誌",
        "違反禁止會車標誌",
        "車輛或機械操作不當(慎)",
        "方向不定(不包括危險駕車)",
        "搶(闖)越平交道",
        "違反禁行車種標誌(字)",
        "違反禁止進入標誌",
        "違反禁止各種車輛進入標誌",
    ],

    "Posture": [
        "違規(臨時)停車",
        "開啟或關閉車門不當",
        "未待乘客安全上下而開車",
        "車輛拋錨未採安全措施",
        "發生事故後，未採取安全措施",
        "暗處停車無燈光、標識",
        "車輛未停妥滑動致生事故",
        "停車操作時未注意安全",
        "上下車輛時未注意安全",
        "未待車輛停妥而上下車",
    ],

    # 酒駕 / 疲勞
    "Driver Impairment": [
        "患病或服用藥物(疲勞)駕駛",
        "酒醉(後)駕駛",
        "打瞌睡或疲勞駕駛",
        "吸食違禁物駕駛",
        "打瞌睡或疲勞駕駛(包括連續駕車8小時)",
    ],

    # 其他
    "Other": [
        "在道路上嬉戲或奔走不定",
        "其他引起事故之疏失或行為",
        "峻狹坡路會車，下坡車未讓上坡車先行",
        "使用車輛自動駕駛或先進駕駛輔助系統設備(裝置)不符規定",
        "使用自動駕駛或先進駕駛輔助系統設備不符規定",
        "乘坐不當(慎)",
        "危險駕駛",
    ],

    # 車輛
    "Vehicle":[
        "車輪脫落或輪胎爆裂",
        "車輛零件脫落",
        "其他機件失靈或故障",
        "煞車失靈或故障",
        "裝載貨物不穩妥",
        "裝載未盡安全措施",
        "物品滾(滑行)或飛(掉)落",
        "車輛附屬機具或車門未盡安全措施",
        "方向操縱系統故障",
        "超載人員",
        "載運貨物超重",
        "裝卸貨物不當",
        "其他裝載不當",
        "燈光系統故障",
        "載運貨物超長、寬、高",
        "夜間行駛無燈光設備",
    ],

    # 環境
    "Environmental": [
        "在道路上工作之人員未設適當標識",
        "因光線、視線遮蔽致生事故",
        "道路設施、植栽或其他裝置倒塌或掉落",
        "施工安全防護措施未完善",
        "平交道看守疏失或未放柵欄",
        "強風、暴雨、濃霧(煙)",
        "動物竄出",
        "路況危險無安全(警告)設施",
        "未依法令授權指揮交通或指揮不當",
        "道路設施(備)、植栽或其他裝置，倒塌或掉(斷)落",
        "物品(件)滾(滑行)或飛(掉)落",
        "其他交通管制不當",
        "施工安全防護措施未依規定或未盡完善(備)",
    ],

    # 未發現
    "Unidentified": [
        "尚未發現肇事因素",
        "相關跡證不足且無具體影像紀錄",
        "肇事逃逸未查獲，無法查明肇因",
        "事故發生時當事者逕自離開現場",
        "被車輛輾壓之不明物體彈飛",
        "相關跡證不足且無具體影像紀錄，當事人各執一詞，經分析後無法釐清肇事原因",
    ]
}

####################### This is for model V5
column_translation = {
    # --- 1. 來自 gis_osm_roads_free_1.shp (roads) ---
    'road_len_motorway': '國道(高速公路)長度(roads)',
    'road_len_motorway_link': '國道匝道長度(roads)',
    'road_len_trunk': '快速道路長度(roads)',
    'road_len_trunk_link': '快速道路匝道長度(roads)',
    'road_len_primary': '省道(主要幹道)長度(roads)',
    'road_len_primary_link': '省道匝道長度(roads)',
    'road_len_secondary': '縣道(次要幹道)長度(roads)',
    'road_len_secondary_link': '縣道匝道長度(roads)',
    'road_len_tertiary': '鄉道(一般道路)長度(roads)',
    'road_len_tertiary_link': '鄉道匝道長度(roads)',
    'road_len_unclassified': '無分級道路長度(roads)',
    'road_len_residential': '住宅區街道(巷弄)長度(roads)',
    'road_len_living_street': '人車共用道(生活街道)長度(roads)',
    'road_len_service': '服務道路長度(roads)',
    'road_len_pedestrian': '行人徒步區長度(roads)',
    'road_len_track': '產業道路長度(roads)',
    'road_len_busway': '公車專用道長度(roads)',
    'road_len_cycleway': '自行車道長度(roads)',
    'road_len_footway': '人行道長度(roads)',
    'road_len_path': '小徑長度(roads)',
    'road_len_steps': '階梯長度(roads)',
    'road_len_bridleway': '馬道長度(roads)',
    'road_len_unknown': '未知類型道路長度(roads)',
    
    # 產業道路細分 (roads)
    'road_len_track_grade1': '產業道路_硬鋪面(roads)',
    'road_len_track_grade2': '產業道路_混合鋪面(roads)',
    'road_len_track_grade3': '產業道路_軟混合(roads)',
    'road_len_track_grade4': '產業道路_植被壓實(roads)',
    'road_len_track_grade5': '產業道路_鬆軟泥土(roads)',

    # --- 2. 來自 gis_osm_traffic_free_1.shp (traffic) ---
    'count_traffic_signals': '交通號誌(紅綠燈)數量(traffic)',
    'count_stop': '停車標誌數量(traffic)',
    'count_crossing': '行人穿越道(斑馬線)數量(traffic)',
    'count_speed_camera': '測速照相機數量(traffic)',
    'count_parking': '路邊停車點數量(traffic)',
    'count_motorway_junction': '交流道數量(traffic)',

    # --- 3. 來自 gis_osm_transport_free_1.shp (transport) ---
    'count_bus_stop': '公車站牌數量(transport)',
    'count_train_station': '火車站數量(transport)',

    # --- 4. 來自 gis_osm_pois_free_1.shp (pois) ---
    # 'count_alcohol': '飲酒場所數量(pois)',
    # 'count_convenience': '便利商店數量(pois)',
    # 'count_school': '學校數量(pois)',

    # --- 5. 來自 外部 CSV 資料 (local data) ---
    'count_mrt': '捷運站出口數量(mrt)',
    'count_youbike': 'YouBike站點數量(youbike)',
    'count_parking_official': '公有路外停車場數量(parkinglot)',

    'count_intersection': '交叉口數量(roads)',
    'count_spd_points': '速差點數量(roads)',

    # --- 6. 模型結果 ---
    'gi_category': 'gi_category'
}

####################### This is for model V6

column_priorities = {
    '車道劃分設施-分道設施-快車道或一般車道間名稱': {
        '未繪設車道線': 1,
        '禁止變換車道線(無標記)': 1,
        '禁止變換車道線(附標記)': 1,
        '車道線(無標記)': 1,
        '車道線(附標記)': 1
    },
    '車道劃分設施-分道設施-快慢車道間名稱': {
        '未繪設快慢車道分隔線': 1,
        '快慢車道分隔線': 1,
        '窄式快慢車道分隔島(無柵欄)': 1,
        '窄式快慢車道分隔島(附柵欄)': 1,
        '寬式快慢車道分隔島(50公分以上)': 1
    },
    '車道劃分設施-分道設施-路面邊線名稱': {
        '無': 1,
        '有': 1
    },

    '車道劃分設施-分向設施大類別名稱': {
        '無': 1,
        '行車分向線': 1,
        '雙向禁止超車線': 1,
        '單向禁止超車線': 1,
        '中央分向島': 1
    },
    '事故類型及型態子類別名稱': {
        '對撞': 1,
        '衝出路外': 1,
        '路口交岔撞': 1,
        '穿越道路中': 1,
        '撞路樹': 1,
        '撞護欄(樁)': 1,
        '自撞': 1,
        '追撞': 1,
        '其他': 1
    },
    '道路型態子類別名稱': {
        '彎曲路及附近': 1,
        '坡路': 1,
        '多岔路': 1,
        '三岔路': 1,
        '四岔路': 1,
        '圓環': 1,
        '隧道': 1,
        '涵洞': 1,
        '橋樑': 1,
        '高架道路': 1,
        '直路': 1,
        '其他': 1
    },
    '號誌-號誌種類名稱': {
        '無號誌': 1,
        '閃光號誌': 1,
        '行車管制號誌': 1,
        '行車管制號誌(附設行人專用號誌)': 1
    }
}