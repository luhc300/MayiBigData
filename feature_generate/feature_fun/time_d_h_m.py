def time_split(s_mall):
    import pandas as pd
    import numpy as np
    import re
    s_mall_data = pd.read_csv('single_mall/%s.csv' % s_mall,header=0)
    s_mall_data.columns = ['row_id','user_id','shop_id','time_stamp','longitude_user','latitude_user','wifi_infos','category_id','longitude_shop','latitude_shop','price','mall_id']
    s_mall_data_time=s_mall_data[['time_stamp']]
    s_mall_data_time['date']=np.nan
    s_mall_data_time['hour']=np.nan
    s_mall_data_time['minute']=np.nan
    for index_time in range(0,len(s_mall_data_time)):
        a1_time=s_mall_data_time.iloc[index_time,0]
        a2_time=re.split(' ',a1_time)
        a3_time=re.split('-',a2_time[0])
        s_mall_data_time.loc[index_time,'date']=a3_time[2]
        a4_time=re.split(':',a2_time[1])
        s_mall_data_time.loc[index_time,'hour']=a4_time[0]
        s_mall_data_time.loc[index_time,'minute']=a4_time[1]
    s_mall_data_time_d_h_m=s_mall_data_time[['date','hour','minute']]
    return s_mall_data_time_d_h_m