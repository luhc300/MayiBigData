"""
Created by Haochuan Lu on 10/24/17.
"""
import csv as csv
import numpy as np
import pandas as pd
from tpot import TPOTClassifier

# 将特征分别读为features，labels
def tpot_train(name):
    #features = np.loadtxt("data/second/" + name + "_f.csv", delimiter=",", skiprows=1)
    #labels = np.loadtxt("data/second/" + name + "_l.csv", delimiter=",", skiprows=1)
    features_df = pd.read_csv("data/third/" + name + "_f.csv")
    labels_df = pd.read_csv("data/third" + name + "_l.csv")
    features = np.array(features_df)
    labels = np.array(labels_df).reshape(-1)


    tpot_config = {
        'sklearn.ensemble.RandomForestClassifier': {
            'n_estimators': [100],
            'criterion': ["gini", "entropy"],
            'max_features': np.arange(0.05, 1.01, 0.05),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'bootstrap': [True, False]
        },

        'sklearn.preprocessing.Binarizer': {
            'threshold': np.arange(0.0, 1.01, 0.05)
        },

        'sklearn.decomposition.FastICA': {
            'tol': np.arange(0.0, 1.01, 0.05)
        },

        'sklearn.cluster.FeatureAgglomeration': {
            'linkage': ['ward', 'complete', 'average'],
            'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
        },

        'sklearn.preprocessing.MaxAbsScaler': {
        },

        'sklearn.preprocessing.MinMaxScaler': {
        },

        'sklearn.preprocessing.Normalizer': {
            'norm': ['l1', 'l2', 'max']
        },

        'sklearn.kernel_approximation.Nystroem': {
            'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
            'gamma': np.arange(0.0, 1.01, 0.05),
            'n_components': range(1, 11)
        },

        'sklearn.decomposition.PCA': {
            'svd_solver': ['randomized'],
            'iterated_power': range(1, 11)
        },

        'sklearn.preprocessing.PolynomialFeatures': {
            'degree': [2],
            'include_bias': [False],
            'interaction_only': [False]
        },

        'sklearn.kernel_approximation.RBFSampler': {
            'gamma': np.arange(0.0, 1.01, 0.05)
        },

        'sklearn.preprocessing.RobustScaler': {
        },

        'sklearn.preprocessing.StandardScaler': {
        },

        'tpot.builtins.ZeroCount': {
        },

        # Selectors
        'sklearn.feature_selection.SelectFwe': {
            'alpha': np.arange(0, 0.05, 0.001),
            'score_func': {
                'sklearn.feature_selection.f_classif': None
            }
        },

        'sklearn.feature_selection.SelectPercentile': {
            'percentile': range(1, 100),
            'score_func': {
                'sklearn.feature_selection.f_classif': None
            }
        },

        'sklearn.feature_selection.VarianceThreshold': {
            'threshold': np.arange(0.05, 1.01, 0.05)
        },

        'sklearn.feature_selection.RFE': {
            'step': np.arange(0.05, 1.01, 0.05),
            'estimator': {
                'sklearn.ensemble.ExtraTreesClassifier': {
                    'n_estimators': [100],
                    'criterion': ['gini', 'entropy'],
                    'max_features': np.arange(0.05, 1.01, 0.05)
                }
            }
        },

        'sklearn.feature_selection.SelectFromModel': {
            'threshold': np.arange(0, 1.01, 0.05),
            'estimator': {
                'sklearn.ensemble.ExtraTreesClassifier': {
                    'n_estimators': [100],
                    'criterion': ['gini', 'entropy'],
                    'max_features': np.arange(0.05, 1.01, 0.05)
                }
            }
        }

    }

    tpot = TPOTClassifier(generations=2, population_size=10, verbosity=2,
                          config_dict=tpot_config)
    tpot.fit(features, labels)

    tpot.export('data/result/20171024/script/' + name + '.py')

    T_features = pd.read_csv("可改.csv")
    prediction = tpot.predict(T_features)
    mapping = pd.read_csv("data/second/mapping/" + name + "_map.csv")
    mapped_prediction = mapping["shop_id"].iloc[prediction]
    result = pd.DataFrame(T_features["row_id"])
    result["shop_id"] = np.array(mapped_prediction)
    return result


if __name__ == "__main__":
    flag = True
    mall_names = pd.read_csv("data/mall_name.csv")
    origin = pd.read_csv("data/evaluation_public.csv")
    # print(mall_names)
    name_list = list(np.array(mall_names).reshape(-1))
    print(name_list)
    result = None
    i = 0
    for name in name_list:
        i += 1
        print(i)
        if flag:
            flag = False
            result = tpot_train(name)
        else:
            result = result.append(tpot_train(name))
    result_final = pd.merge(origin, result, on=["row_id"])
    result_final = result_final[["row_id","shop_id"]]
    result_final.to_csv("data/result/result_20171020.csv", index=False)