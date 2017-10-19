def wifi_connetion(s_mall):
    import pandas as pd
    import numpy as np
    import re
    s_mall_data = pd.read_csv('single_mall/%s.csv' % s_mall,header=0)
    s_mall_data.columns = ['row_id','user_id','shop_id','time_stamp','longitude_user','latitude_user','wifi_infos','category_id','longitude_shop','latitude_shop','price','mall_id']
    s_mall_data_wifi=s_mall_data[['wifi_infos']]
    s_mall_data_wifi['wifi_connetion']=np.nan
    q_wifi = []
    for i_wifi_statistic in range(0,len(s_mall_data_wifi)):
        a1=s_mall_data_wifi.iloc[i_wifi_statistic,0]
        a2=re.split(';',a1)
        for i in range(0,len(a2)):
            a3=re.split('\\|',a2[i])
            q_wifi.append(a3[0])
    len(q_wifi)
    wifi_one=pd.DataFrame(np.array(q_wifi))
    wifi_one.drop_duplicates(inplace=True)
    from sklearn import preprocessing
    le_wifi = preprocessing.LabelEncoder()
    le_wifi.fit(wifi_one)
    for index_connet in range(0,len(s_mall_data_wifi)):
        a1=s_mall_data_wifi.iloc[index_connet,0]
        a2=re.split(';',a1)
        for i in range(0,len(a2)):
            a3=re.split('\\|',a2[i])
            if a3[2]=='true':
                s_mall_data_wifi.loc[index_connet,'wifi_connetion']=le_wifi.transform([a3[0]])[0]
    w1=s_mall_data_wifi[['wifi_connetion']]
    w16=w1.replace(np.nan,0)
    return w16