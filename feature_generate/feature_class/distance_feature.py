import feature_generate.feature_class.feature
from math import *
from feature_generate.feature_class.feature import Feature

class DistanceFeature(Feature):
    def __init__(self):
        Feature.__init__(self)

    def generate(self, mall_name):
        import pandas as pd
        import numpy as np

        s_mall_data = pd.read_csv(self._data_home+'single_mall/%s.csv' % mall_name, header=0)
        s_mall_data.columns = ['row_id', 'user_id', 'shop_id', 'time_stamp', 'longitude_user', 'latitude_user',
                               'wifi_infos', 'category_id', 'longitude_shop', 'latitude_shop', 'price', 'mall_id']
        s_mall_data_longitude_and_latitude = s_mall_data[
            ['longitude_user', 'latitude_user', 'longitude_shop', 'latitude_shop']]

        shop_2L = s_mall_data[['shop_id', 'longitude_shop', 'latitude_shop']]
        shop_2L.drop_duplicates(inplace=True)

        np_shop_2L = np.array(shop_2L)
        pd_shop_2L = pd.DataFrame(np_shop_2L, columns=['shop_id', 'longitude_shop', 'latitude_shop'])

        np_shop = np.array(shop_2L['shop_id'])

        np_build = np.array([np.NaN] * len(s_mall_data_longitude_and_latitude) * len(np_shop)).reshape(
            (len(s_mall_data_longitude_and_latitude), len(np_shop)))

        pd_build = pd.DataFrame(np_build, columns=np_shop)

        for i_all_shop in range(0, len(np_shop)):
            user_long = s_mall_data_longitude_and_latitude.longitude_user
            user_lat = s_mall_data_longitude_and_latitude.latitude_user
            shop_long = pd_shop_2L.loc[i_all_shop]['longitude_shop']
            shop_lat = pd_shop_2L.loc[i_all_shop]['latitude_shop']
            for i_dis in range(0, len(s_mall_data_longitude_and_latitude)):
                pd_build.iloc[i_dis, i_all_shop] = self.__calcDistance(user_long[i_dis], user_lat[i_dis], shop_long, shop_lat)

        return pd_build


    def __calcDistance(self,Lat_A, Lng_A, Lat_B, Lng_B):


        ra = 6378.140  # 赤道半径 (km)
        rb = 6356.755  # 极半径 (km)
        flatten = (ra - rb) / ra  # 地球扁率
        rad_lat_A = radians(Lat_A)
        rad_lng_A = radians(Lng_A)
        rad_lat_B = radians(Lat_B)
        rad_lng_B = radians(Lng_B)
        pA = atan(rb / ra * tan(rad_lat_A))
        pB = atan(rb / ra * tan(rad_lat_B))
        xx = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B))
        c1 = (sin(xx) - xx) * (sin(pA) + sin(pB)) ** 2 / cos(xx / 2) ** 2
        c2 = (sin(xx) + xx) * (sin(pA) - sin(pB)) ** 2 / sin(xx / 2) ** 2
        dr = flatten / 8 * (c1 - c2)
        distance = ra * (xx + dr)
        return distance

if __name__ == "__main__":
    result = DistanceFeature().generate("m_6167")
    print(result)
    result.to_csv("D:/Programs/Python/MayiBigData/data/single_mall_generated_feature/m_6167_2.csv")