import matplotlib.pyplot as plt
from config import countycity_dct
import numpy as np
import pandas as pd

def speed_bin(data):
    max_speed = data['速限-第1當事者'].max()
    bins = range(0, int(max_speed) + 11, 10)

    data['速限-第1當事者'] = pd.cut(
        data['速限-第1當事者'],
        bins=bins,
        right=False, 
        include_lowest=True,
        labels=[f"{i}-{i+9}" for i in bins[:-1]]
    )

    return data['速限-第1當事者']

def youbike_bin(x):
    if x >= 5:
        return '5+'
    elif x >= 3:
        return '3~4'
    elif x >= 1:
        return '1~2'
    else:
        return '0'

class BicycleFacilityAnalyzer:
    def __init__(self, result, countycity_dct, countys_lst=None):
        self.result = result
        self.countycity_dct = countycity_dct
        self.facility_ratio_df = None
        self.countys_lst = countys_lst

    def get_facility_ration(self, cols='當事者區分-類別-子類別名稱-車種', attr='腳踏自行車'):
        result = self.result[self.result['COUNTYNAME'].isin(self.countys_lst)]
        youbike_type = 'youbike_100m_count'

        include_youbike = result[result[cols] == attr]

        result['youbike_bin'] = result[youbike_type].apply(youbike_bin)
        include_youbike['youbike_bin'] = include_youbike[youbike_type].apply(youbike_bin)

        total_accidents_by_facility = result.groupby(['COUNTYNAME', 'youbike_bin']).size().reset_index(name='total')
        bike_accidents_by_facility = include_youbike.groupby(['COUNTYNAME', 'youbike_bin']).size().reset_index(name='bike')

        facility_ratio_df = pd.merge(total_accidents_by_facility, bike_accidents_by_facility, 
                                    on=['COUNTYNAME', 'youbike_bin'], how='left').fillna(0)

        facility_ratio_df['bike_accident_ratio'] = facility_ratio_df['bike'] / facility_ratio_df['total']
        facility_ratio_df = facility_ratio_df[facility_ratio_df['bike'] > 0]
        facility_ratio_df = facility_ratio_df[['COUNTYNAME', 'youbike_bin', 'bike_accident_ratio', 'total', 'bike']]
        facility_ratio_df = facility_ratio_df.sort_values(['COUNTYNAME', 'youbike_bin'])
        facility_ratio_df['COUNTYNAME'] = facility_ratio_df['COUNTYNAME'].map(self.countycity_dct)

        self.facility_ratio_df = facility_ratio_df

    def plot_facility_accident_ratio(self, ylabel='Bicycle Accident Ratio'):
        bins = ['0', '1~2', '3~4', '5+']
        countys = self.facility_ratio_df['COUNTYNAME'].unique()
        bar_width = 0.2
        x = np.arange(len(countys))

        colors = ["#479560", "#7D9A8D", "#B2C2B2", "#C4B2B2"]

        plt.figure(figsize=(10, 6))

        for i, bin_label in enumerate(bins):
            df_bin = self.facility_ratio_df[self.facility_ratio_df['youbike_bin'] == bin_label].set_index('COUNTYNAME').reindex(countys)
            ratios = df_bin['bike_accident_ratio']
            totals = df_bin['total']
            bars = plt.bar(x + i * bar_width, ratios, width=bar_width, label=bin_label, color=colors[i])
            for bar, total in zip(bars, totals):
                if not np.isnan(bar.get_height()):
                    plt.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height(),
                        str(int(total)) if not np.isnan(total) else "",
                        ha='center', va='bottom', fontsize=9
                    )

        plt.xlabel('County')
        plt.ylabel(ylabel)
        plt.xticks(x + bar_width * (len(bins)-1)/2, countys)
        plt.legend(title='YouBike Facilities')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # plt.savefig(f'../ComputedDataV2/plot/{ylabel}.pdf')
