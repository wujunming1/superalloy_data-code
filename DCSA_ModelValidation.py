'''
65.89	5.61	0.99	5.2	0	7.56	0.81	4.27	8.52	0.07	0.03	0	0.83	0.22	4	4	20	1280	1080	871	1040	137.2	61.06426846	4.51733E-21	53.96456487	0.350781578	0.75214	454.3
58.832	1.26	12.4	4.35	0	6.32	0.73	6.6	7.68	0.062	0.026	0	1.5	0.24	4	4	20	1270	1080	871	1040	137.2	59.77759412	2.24667E-21	57.57985457	0.357581635	0.54641	211.1
64.53	0	5	4.15	3.68	6.5	0.92	10.6	4.42	0	0	0	0	0.2	3	4	20	1276	1080	871	204	759	41.04733707	1.80716E-21	69.04909109	0.349344299	0.60239	256.2
64.73	0	5	4.15	3.68	6.5	0.92	10.6	4.42	0	0	0	0	0	3	4	20	1288	1080	871	315	448.5	41.34221675	1.79973E-21	69.20082315	0.349701573	0.5966	277.1
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import numpy as np
df = pd.read_excel("test_data_file/6-new validated instance.xlsx")
validation_data = df.values#选取最后四条验证样本
# print(validation_data)
data, target = validation_data[4:5, :-1], validation_data[4:5, -1]
# print("22222", target[0:2])
data, target = MinMaxScaler().fit_transform(data), np.log(target)
clf = joblib.load("model_for_custers/cluster_7RF.model")
print("22222", target)
print(np.exp(target), np.exp(clf.predict(data)))
