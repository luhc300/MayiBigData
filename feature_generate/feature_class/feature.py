from abc import ABCMeta, abstractmethod
from configs.path_config import DATA_PATH_HOME

class Feature(metaclass=ABCMeta):

    def __init__(self):
        self._data_home = DATA_PATH_HOME
        pass

    @abstractmethod
    def generate(self, mall_name):
        pass

    def get_feature_name(self):
        return self.__class__.__name__