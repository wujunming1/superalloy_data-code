import numpy as np
import pandas as pd
import sys, os


def load_data(input_file, sheet_name):
    source_data = pd.read_excel(io=input_file, sheetname=sheet_name)
    # source_data = pd.read_excel(io=input_file)
    np_data = source_data.as_matrix()
    # print(source_data.columns)
    return source_data, np_data


def data_process(array_data):
    from sklearn import preprocessing
    for i in range(len(array_data[0, :])-2):
        array_data[:, i+1] = preprocessing.scale(array_data[:, i+1])
    # array_data = preprocessing.StandardScaler().fit_transform(array_data)
    # array_data[array_data == 0] = 0.0001
    # array_data[:, 21] = array_data[:, 21]/100
    # array_data[:, 22] = array_data[:, 22] / 100
    # array_data[:, -1] = np.log(array_data[:, -1])
    # print(array_data.shape)


def test_write_excel(test):
    r1 = pd.DataFrame(test)
    r = pd.concat([r1])
    r.to_excel('test.xls')


def data_split(array_data, random_seed):
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(array_data[:, :-1], array_data[:, -1],
                                                        test_size=0.25, random_state=random_seed)
    return train_x, test_x, train_y, test_y


def gaussian_model():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, RBF
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    # parameter = "C(1, (1e-4, 10000)) * RationalQuadratic(alpha=0.01, length_scale_bounds=(1e-5, 20))"
    parameter = ' gaussian_model '
    # kernel_5 = C(1, (1e-4, 1)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(0.01, 2000))
    kernel = C(1, (0.01, 10)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(0.1, 2000))
    model = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=10)
    return model, parameter


def linear_model():
    from sklearn.linear_model import LinearRegression
    parameter = ' Linear_model '
    model = LinearRegression()

    return model, parameter


def svr_model():
    from sklearn.svm import SVR
    parameter = ' svr_model '
    model = SVR(kernel='rbf', C=100, gamma='auto')
    return model, parameter


def forest_model():
    from sklearn.ensemble import RandomForestRegressor
    parameter = ' RandomForest_model '
    # model = RandomForestRegressor(n_estimators=15, max_depth=4, criterion='mae', bootstrap=True)
    model = RandomForestRegressor(n_estimators=15, max_depth=4, criterion='mae', bootstrap=True)
    return model, parameter


def adaboost_model():
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    parameter = 'adaboostRegressor '
    model = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=100, loss='exponential')
    return model, parameter


def poly_model():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    parameter = 'poly '
    model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                      ('linear', LinearRegression(fit_intercept=False))])
    return model, parameter


def network_model():
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(39, input_dim=27, activation='relu'))
    model.add(Dense(1,activation='linear' ))
    model.compile(loss="MAPE", optimizer='adam', metrics=['accuracy'])
    parameter = 'ANN'
    return model, parameter


def train_model(train_x, test_x, train_y, test_y, select_model):
    # print(train_x.shape)
    if select_model == 0:
        model, parameter = forest_model()
        model.fit(train_x, train_y)
    elif select_model == 1:
        model, parameter = svr_model()
        model.fit(train_x, train_y)
    # elif select_model == 2:
    #     model, parameter = adaboost_model()
    # elif select_model == 3:
    #     model, parameter = linear_model()
    #     model.fit(train_x, train_y)
    # elif select_model == 4:
    #     model, parameter = network_model()
    #     model.fit(train_x, train_y, batch_size=2, epochs=300)
    else:
        model, parameter = gaussian_model()
        model.fit(train_x, train_y)
            # 训练模型
    # model.get_params()
    # print('系数:', model.get_params())
    predict_y = model.predict(test_x)   # 预测模型

    true_err = predict_y - test_y
    absolute_err = abs(true_err)
    # print(parameter, 'test_y is : ', test_y, '\npredict_y is : ', predict_y, '\nerr_percent is: ',
    #       sum(absolute_err/test_y)/len(test_y))
    return predict_y, true_err, parameter


def train_tree(source_data, train_x, test_x, train_y, test_y, select_model):
    from sklearn import tree
    import pydotplus
    model, parameter = forest_model()
    model.fit(train_x, train_y)  # 训练模型

    Estimators = model.estimators_
    for num in range(15):
        dot_data = tree.export_graphviz(Estimators[num], out_file=None, filled=True, rounded=True,
                                        feature_names=source_data.columns[1:-1], special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png("C:\\Users\\15769\\Desktop\\result\\tree\\tree"+str(select_model)+"-"+str(num)+".png")


    predict_y = model.predict(test_x)  # 预测模型
    true_err = predict_y - test_y
    absolute_err = abs(true_err)
    print(parameter, 'test_y is : ', test_y, '\npredict_y is : ', predict_y, '\nerr_percent is: ',
          sum(absolute_err / test_y) / len(test_y))
    return predict_y, true_err, parameter


def plot_out(predict_y, test_y, title, save_path):
    import matplotlib.pyplot as plt
    # print(predict_y.shape)
    min_y = min(predict_y)
    max_y = max(predict_y)
    min_x = min(test_y)
    max_x = max(test_y)
    plt.ylim(ymax=max(max_x, max_y), ymin=min(min_x, min_y))
    plt.xlim(xmax=max(max_x, max_y), xmin=min(min_x, min_y))
    min_value = min(min_x, min_y)
    max_value = max(max_x, max_y)
    plt.scatter(test_y, predict_y, linewidths=True, vmax=1)
    # plt.plot([min_value, max_value], [min_value, max_value], "--")
    plt.xlim(0, 3)
    plt.ylim(0, 3)
    plt.plot([0, 3], [0, 3], "--")
    plt.xlabel('real_life')
    plt.ylabel('predict_lift')
    plt.title(title)
    plt.savefig("dbscan\\"+title+".svg", format='svg')
    plt.show()


def write_excel(source_data, array_data, predict_matrix, parameter, category):
    source_life = array_data[:, -1]
    columns_1 = pd.DataFrame(array_data)
    columns_2 = pd.DataFrame(predict_matrix)
    average_life, average_err, average_err_percent, average_err_suqare= calculate_indicator(array_data[:, -1], predict_matrix)
    columns_3 = pd.DataFrame(array_data[:, -1])
    columns_4 = pd.DataFrame(average_life)
    columns_5 = pd.DataFrame(average_err)
    columns_6 = pd.DataFrame(average_err_percent)
    columns_7 = pd.DataFrame(average_err_suqare)
    tags = []
    for i in range(len(predict_matrix[0, :])):
        tags.append('times_' + str(i))
    columns_all = pd.concat([columns_1, columns_2, columns_3, columns_4, columns_5, columns_6, columns_7], axis=1)
    columns_all.columns = list(source_data.columns) + list(tags) + ['source_life', 'average_life',
                                                                    'average_err', 'average_err_percent', 'square']
    print('the'+parameter+' model err_percent is :' + str(sum(average_err_percent)/len(average_err_percent)))
    columns_all.to_excel('brg\\' + category + parameter + '.xls')
    plot_out(average_life, source_life, parameter + 'MAPE: ' +
             str(sum(average_err_percent) / len(average_err_percent)), 'cluster_result')
    return sum(average_err_percent)/len(average_err_percent)


def write_plot(source_data, test_x, test_y, predict_y, path1):
    r7 = pd.DataFrame(np.arange(0,len(test_y)).T)
    r1 = pd.DataFrame(test_x)
    r0 = pd.DataFrame(test_y)
    r2 = pd.DataFrame(predict_y)
    r3 = pd.DataFrame(predict_y - test_y)
    r4 = pd.DataFrame(abs(predict_y - test_y)/test_y)
    # r = pd.concat([r7, r1, r0, r2, r3, r4], axis=1)
    # r.columns = list(source_data.columns[1:]) + ['source_y','predict_y', 'err', 'err_percent']
    r = pd.concat([r0, r2, r3, r4],axis=1)
    r.columns = ['source_y','predict_y', 'err', 'err_percent']
    # 设定存储地址
    r.to_excel(os.path.join(path1, 'test.xls'))
    plot_out(test_y, predict_y, 'title', path1)


def calculate_indicator(source_life, predict_matrix):
    row_num = (len(predict_matrix[:, 0]))
    average_life = np.zeros((row_num, 1))      # 构造平均列
    average_err = np.zeros((row_num, 1))
    average_err_percent = np.zeros((row_num, 1))
    average_err_square = np.zeros((row_num, 1))

    for i in range(row_num):
        average_life[i] = sum([x for x in predict_matrix[i, :] if x > 0]) / (predict_matrix[i, :] > 0).sum()
        average_err[i] = abs(source_life[i]) - abs(average_life[i])
        average_err_percent[i] = (abs(average_err[i]))/(abs(source_life[i]))
        average_err_square[i] = average_err[i]*average_err[i]
        # print(i)
    # print(average_life.shape, average_err.shape, average_err_percent.shape)
    print('avg_err is :' + str(np.average(average_err_percent)))
    return average_life, average_err, average_err_percent, average_err_square


def run():
    # C:\\Users\\15769\\Desktop\\result\\cluster_condition_center.xls
    result_save = np.zeros((8,3))

    for i in range(1):
        sheet_name = u'cluster_'+str(i)
        source_data, array_data = load_data("brg\\cluster_data2.xls",
                                            sheet_name=sheet_name)
        data_process(array_data)
        for select_model in range(3):
            # 这里的次数和测试样本大小有关，不能选的太小，否则会造成最后有的样本没有被选中
            times_num = 100
            predict_matrix = np.full((len(array_data[:, 0]), times_num), 0, dtype=float)
            for times in range(times_num):
                # print('\n\n this is the round: ' + str(times))
                train_x, test_x, train_y, test_y = data_split(array_data, times+100)
                testX = test_x
                testY = test_y
                # testX = train_x
                # testY = train_y
                # select_model = 2
                predict_y, one_err, parameter = train_model(train_x[:, 1:], testX[:, 1:], train_y, testY, select_model)
                # print(testX)
                # predict_y, one_err, parameter = train_tree(source_data,train_x[:, 1:], testX[:, 1:], train_y, testY, times)
                for k in range(len(testY)):
                    location = testX[k, 0]
                    # print(location, k, predict_y[k], predict_matrix.shape, times)
                    # 下标要用int型
                    predict_matrix[int(location-1), times] = predict_y[k]
            # print('predict_matrix: ', predict_matrix)
            once_MAPE = write_excel(source_data, array_data, predict_matrix, sheet_name + parameter, 'gbr')
            result_save[i][select_model] = once_MAPE
        print(sheet_name, " the sample length is ", len(array_data[:, 0]))
    print(result_save)


def run_one(path):
    source_data, array_data = load_data('\\cluster_result\\多尺度样本聚类.xls', 'cluster_1')
    data_process(array_data)

    # 这里的次数和测试样本大小有关，不能选的太小，否则会造成最后有的样本没有被选中

    train_x, test_x, train_y, test_y = data_split(array_data, 100)
    print('test_x is', test_x)
    testX = test_x
    testY = test_y
    predict_y, one_err, parameter = train_model(train_x, testX, train_y, testY, 1)
    # write_excel(source_data, array_data, predict_y, parameter, 'cluster_3')
    write_plot(source_data, test_x, test_y, predict_y, path)


if __name__ == "__main__":
    # path = 'C:\\Users\\15769\\Desktop\\result'
    # parameterlist = []
    # for i in range(1, len(sys.argv)):
    #     para = sys.argv[i]
    #     parameterlist.append(para)
    # print(parameterlist)
    # run_one(path,parameterlist[0], int(parameterlist[1]), float(parameterlist[2]) )
    run()
    # run_one(path)
    # run()