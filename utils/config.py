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

    '事故類型及型態子類別名稱': 'Accident type and pattern (subcategory)',
    '車道劃分設施-分向設施子類別名稱': 'Lane division facility - directional separation (subcategory)',
    '道路型態子類別名稱': 'Road type (subcategory)',

    '速限-第1當事者': 'Speed limit (Party 1)',
    '道路類別-第1當事者-名稱': 'Road category (Party 1)',

    '肇因研判子類別名稱-主要': 'Primary cause determination (subcategory)'
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
    }
}

countycity_dct = {
    '臺南市': 'Tainan',
    '高雄市': 'Kaohsiung',
    '嘉義市': 'Chiayi City',
    '嘉義縣': 'Chiayi County',
    '屏東縣': 'Pingtung County',
    '彰化縣': 'Changhua County',
    '雲林縣': 'Yunlin County',
    '臺中市': 'Taichung',
    '苗栗縣': 'Miaoli County',
    '新竹市': 'Hsinchu City',
    '新竹縣': 'Hsinchu County',
    '臺東縣': 'Taitung County',
    '桃園市': 'Taoyuan',
    '新北市': 'New Taipei',
    '臺北市': 'Taipei',
    '花蓮縣': 'Hualien County',
    '宜蘭縣': 'Yilan County'
}
