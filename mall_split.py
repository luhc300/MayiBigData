import pandas as pd
import numpy as np


shop_info = pd.read_csv("data/ccf_first_round_shop_info.csv")
user_shop_behavior = pd.read_csv("data/ccf_first_round_user_shop_behavior.csv")
train_data = pd.merge(user_shop_behavior, shop_info, on='shop_id', how='left')
train_data.columns = ['user_id', 'shop_id', 'time_stamp', 'longitude_user', 'latitude_user', 'wifi_infos',
                          'category_id', 'longitude_shop', 'latitude_shop', 'price', 'mall_id']
mall = train_data["mall_id"]
mall_name = mall.drop_duplicates()
mall_name = mall_name.reset_index()
mall_name = mall_name.drop(["index"], axis=1)
mall_name.to_csv("data/mall_name.csv",index=False)
'''
#print(mall_name)
mall_name_list = list(np.array(mall_name).reshape(-1))
print(mall_name_list)
for name in mall_name_list:
    single_mall_data = train_data[train_data["mall_id"] == name]
    single_mall_data.to_csv("data/single_mall/" + name + ".csv")
'''