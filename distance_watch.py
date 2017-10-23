import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def input(path_feature, path_label):
    feature = pd.read_csv(path_feature, index_col=0)
    length = feature.shape[1]
    distance = feature
    distance = 1 - distance
    label = pd.read_csv(path_label)
    distance["label"] = label
    return distance

def watch(wifi, shop_id, mall_id):
    wifi_selected = wifi[wifi["label"] == shop_id]
    wifi_selected_info = wifi_selected.drop(["label"],axis=1)
    plt.figure()
    for i in range(wifi_selected_info.shape[0]):

        wifi_selected_info_line = wifi_selected_info.iloc[i]
        wifi_selected_info_line.plot()
    path = "data/single_mall_generated_feature/distance/" + mall_id
    if not os.path.exists(path):  ###判断文件是否存在，返回布尔值
        os.makedirs(path)
    path +=  '/' + str(shop_id) + '.png'
    plt.savefig(path)
    plt.close()

name = "m_6167"
path_feature = "data/single_mall_generated_feature/" + name + "_2.csv"
path_label = "data/second/" + name + "_l.csv"
distance = input(path_feature, path_label)
print(distance)

max_label = distance["label"].value_counts(sort=False).reset_index().index.max()
for i in range(max_label+1):
    watch(distance, shop_id=i, mall_id= name)