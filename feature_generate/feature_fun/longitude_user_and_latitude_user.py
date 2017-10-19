def longitude_user_and_latitude_user(s_mall):
    import pandas as pd
    import numpy as np
    s_mall_data = pd.read_csv('single_mall/%s.csv' % s_mall,header=0)
    s_mall_data.columns = ['row_id','user_id','shop_id','time_stamp','longitude_user','latitude_user','wifi_infos','category_id','longitude_shop','latitude_shop','price','mall_id']
    s_mall_data_longitude_and_latitude=s_mall_data[['longitude_user','latitude_user',]]
    return s_mall_data_longitude_and_latitude