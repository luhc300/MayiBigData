import pandas as pd
import numpy as np

shop_info = pd.read_csv("data/ccf_first_round_shop_info.csv")
mall = shop_info["mall_id"]
mall_name = mall.drop_duplicates()
mall_name = mall_name.reset_index()
mall_name = mall_name.drop(["index"], axis=1)

mall_name_list = list(np.array(mall_name).reshape(-1))
print(mall_name_list)
for name in mall_name_list:
    single_mall_data = shop_info[shop_info["mall_id"] == name]
    single_mall_data.to_csv("data/single_mall_shop/" + name + ".csv", index=False)