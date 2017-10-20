import pandas as pd

from feature_generate.feature_class.feature import Feature


class UserPositionFeature(Feature):
    def __init__(self):
        Feature.__init__(self)

    def generate(self, mall_name):
        s_mall_data = pd.read_csv(self._data_home+'single_mall/%s.csv' % mall_name, header=0)
        s_mall_data.columns = ['row_id', 'user_id', 'shop_id', 'time_stamp', 'longitude_user', 'latitude_user',
                               'wifi_infos', 'category_id', 'longitude_shop', 'latitude_shop', 'price', 'mall_id']
        s_mall_data_longitude_and_latitude = s_mall_data[['longitude_user', 'latitude_user']]

        return s_mall_data_longitude_and_latitude