from feature_generate.feature_class.wifi_strength_feature import WifiStrengthFeature
from feature_generate.feature_class.wifi_connection_feature import WifiConnectionFeature
from feature_generate.feature_class.user_position_feature import UserPositionFeature
from feature_generate.feature_class.time_feature import TimeFeature
from feature_generate.feature_class.distance_feature import DistanceFeature
import pandas as pd
import numpy as np
from configs.path_config import DATA_PATH_HOME


class FeatureGenerator:
    def __init__(self):
        self.__action_list = []

    def set_action_list(self, action_list):
        self.__action_list = action_list

    def append_action_list(self, feature_class):
        self.__action_list.append(feature_class)

    def reset_action_list(self):
        self.__action_list = []

    def generate(self, mall_name, old_path=None):
        result = None
        flag = True
        if old_path is not None:
            result = pd.read_csv(old_path)
            flag = False
        for action in self.__action_list:
            feature = action.generate(mall_name)
            if flag:
                result = feature
                flag = False
            else:
                result = pd.concat([result,feature], axis=1)
            print(action.get_feature_name() + " finished")
        return result


if (__name__ == "__main__"):
    fg = FeatureGenerator()
    #fg.set_action_list([WifiStrengthFeature(),WifiConnectionFeature(), UserPositionFeature(), TimeFeature()])
    #fg.set_action_list([DistanceFeature()])
    fg.set_action_list([DistanceFeature()])
    mall_names = pd.read_csv(DATA_PATH_HOME + "mall_name.csv")
    name_list = list(np.array(mall_names).reshape(-1))
    i = 0
    for name in name_list:
        print(i)
        i+=1
        result = fg.generate(name, old_path=DATA_PATH_HOME + "second/" + name + "_f.csv")
        result.to_csv(DATA_PATH_HOME + "single_mall_generated_feature/20171020/" + name + ".csv",index=False)
