from feature_generate.feature_class.feature import Feature


class WifiStrengthFeature(Feature):
    def __init__(self):
        Feature.__init__(self)

    def generate(self, mall_name):
        import pandas as pd
        import numpy as np
        import re
        s_mall_data = pd.read_csv(self._data_home+'single_mall/%s.csv' % mall_name, header=0)
        s_mall_data.columns = ['row_id', 'user_id', 'shop_id', 'time_stamp', 'longitude_user', 'latitude_user',
                               'wifi_infos', 'category_id', 'longitude_shop', 'latitude_shop', 'price', 'mall_id']
        s_mall_data_wifi = s_mall_data[['wifi_infos']]
        q_wifi = []
        for i_wifi_statistic in range(0, len(s_mall_data_wifi)):
            a1 = s_mall_data_wifi.iloc[i_wifi_statistic, 0]
            a2 = re.split(';', a1)
            for i in range(0, len(a2)):
                a3 = re.split('\\|', a2[i])
                q_wifi.append(a3[0])
        len(q_wifi)
        q2 = set()
        for i in range(0, len(s_mall_data_wifi)):
            a1 = s_mall_data_wifi.iloc[i, 0]
            a2 = re.split(';', a1)
            for i in range(0, len(a2)):
                a3 = re.split('\\|', a2[i])
                q2.add(a3[0])
        len(q2)
        w5 = pd.DataFrame(q_wifi, columns=['wifi'])
        w5['nums'] = 1
        w6 = w5.groupby('wifi').sum().reset_index()
        w6[['nums']].max()
        w8 = w6[(w6.nums > 100)]
        w9 = w8['wifi']
        npw9 = np.array(w9)
        np1 = np.array([np.NaN] * len(w8) * len(s_mall_data)).reshape((len(s_mall_data), len(w8)))
        w10 = pd.DataFrame(np1, columns=npw9)
        aa = s_mall_data[['wifi_infos']]
        for index_aa in range(0, len(aa)):
            a1 = aa.iloc[index_aa, 0]
            a2 = re.split(';', a1)
            for i2 in range(0, len(a2)):
                a3 = re.split('\\|', a2[i2])
                w10.iloc[index_aa][a3[0]] = a3[1]
        w12 = w10.replace(np.nan, -100)
        return w12

