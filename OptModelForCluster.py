'''
Clusters	Selected optimal model	Fitness
1	          RF	                 0.8924
2	          GPR	                 0.9595
3	          GPR	                 0.9376
4	          RF	                 0.9692
5	          RF	                 0.9364
6	          SVR	                 0.9548
7	          SVR	                 0.9791
8	          RF	                 0.9212
'''
import pandas as pd


def gaussian_model():
    #GPR model
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, RBF
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    # parameter = "C(1, (1e-4, 10000)) * RationalQuadratic(alpha=0.01, length_scale_bounds=(1e-5, 20))"
    parameter = ' gaussian_model '
    # kernel_5 = C(1, (1e-4, 1)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(0.01, 2000))
    kernel = C(1, (0.01, 10)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(0.1, 2000))
    model = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=10)
    return model, parameter


def svr_model():
    #SVR model
    from sklearn.svm import SVR
    parameter = ' svr_model '
    model = SVR(kernel='rbf', C=100, gamma='auto')
    return model, parameter


def random_forest_model():
    #random forest model
    from sklearn.ensemble import RandomForestRegressor
    parameter = ' RandomForest_model '
    # model = RandomForestRegressor(n_estimators=15, max_depth=4, criterion='mae', bootstrap=True)
    model = RandomForestRegressor(n_estimators=15, max_depth=4, criterion='mae', bootstrap=True)
    return model, parameter


import pickle
from sklearn.externals import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
all_clusters = ["cluster_0", "cluster_1", "cluster_2",
               "cluster_3", "cluster_4",
               "cluster_5","cluster_6","cluster_7"] #eight different
# alloy clusters with various creep mechanisms
candidate_models = ["RF", "GPR", "SVR"]#three candidate machine learning model
cluster_model = {"cluster_0": "RF", "cluster_1": "GPR","cluster_2": "GPR",
                 "cluster_3": "RF","cluster_4": "RF","cluster_5": "SVR",
                 "cluster_6": "SVR","cluster_7": "RF"}
print("all models start running!")
for cluster in cluster_model:
    print(cluster)
    df = pd.read_excel("data_files/多尺度样本聚类_912.xls",sheet_name=cluster)
    cluster_sample = df.values
    # print("11111", cluster_sample)
    data, target = cluster_sample[:, 1:-1],cluster_sample[:, -1]
    data = MinMaxScaler().fit_transform(data)
    target = np.log(target)
    print(data ,target)
    if cluster_model.get(cluster) == "RF":
        rf_model, rf_para = random_forest_model()
        rf_model.fit(data, target)
        joblib.dump(rf_model, "model_for_custers/"+cluster+"RF"+".model")
    if cluster_model.get(cluster) == "GPR":
        rf_model, rf_para = gaussian_model()
        rf_model.fit(data, target)
        joblib.dump(rf_model, "model_for_custers/"+cluster + "GPR" + ".model")
    if cluster_model.get(cluster) == "SVR":
        rf_model, rf_para = svr_model()
        rf_model.fit(data, target)
        joblib.dump(rf_model, "model_for_custers/"+cluster + "SVR" + ".model")
print("all models have already save!")