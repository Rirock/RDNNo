import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from utils.config import Config
from utils.load_data import Data
from utils.tools import *
from model.SVM_model_tensorflow import train, predict

def draw(config: Config, origin_data: Data, logger, predict_data_norm: np.ndarray):
    # 测试结果 画图
    _, label_data_norm = origin_data.get_test_data()
    # 通过保存的均值和方差还原数据
    label_data = label_data_norm * origin_data.std_test[config.label_in_feature_index] + \
                   origin_data.mean_test[config.label_in_feature_index]   
    predict_data = predict_data_norm * origin_data.std_test[config.label_in_feature_index] + \
                   origin_data.mean_test[config.label_in_feature_index]   
    assert label_data_norm.shape[0]==predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    label_data_d = label_data
    predict_data_d = predict_data
    mse_norm = metrics.mean_squared_error(label_data_d[config.predict_day:], predict_data_d[:-config.predict_day])
    logger.info("The MSE is " + str(mse_norm))
    rmse_norm = metrics.mean_squared_error(label_data_d[config.predict_day:], predict_data_d[:-config.predict_day])**0.5
    logger.info("The RMSE is " + str(rmse_norm))
    mae_norm = metrics.mean_absolute_error(label_data_d[config.predict_day:], predict_data_d[:-config.predict_day])
    logger.info("The MAE is " + str(mae_norm))
    r2_norm = metrics.r2_score(label_data_d[config.predict_day:], predict_data_d[:-config.predict_day])
    logger.info("The R2 is " + str(r2_norm))

    # 保存数据
    label_X = range(len(label_data))
    predict_X = [ x + config.predict_day for x in label_X]
    save_data = list(zip(label_data_d[config.predict_day:], predict_data_d[:-config.predict_day]))
    columns = ['label', 'predict']
    df = pd.DataFrame(columns=columns, data=save_data)
    df.to_csv(config.figure_save_path+"data_"+str(config.train_num)+".csv")

    #if not sys.platform.startswith('linux'):    # 无桌面的Linux下无法输出，如果是有桌面的Linux，如Ubuntu，可去掉这一行
    for i in range(label_column_num):
        plt.figure(i+1)                     # 预测数据绘制
        plt.plot(label_X, label_data[:, i], label='label')
        plt.plot(predict_X, predict_data[:, i], label='predict')
        plt.title("Predict stock {} price with {}".format(label_name[i], config.used_frame))
        logger.info("The predicted stock {} for the next {} day(s) is: ".format(label_name[i], config.predict_day) +
                str(np.squeeze(predict_data[-config.predict_day:, i])))
        if config.do_figure_save:
            plt.savefig(config.figure_save_path+"{}predict_{}_with_{}.png".format(config.continue_flag, label_name[i], config.used_frame))
    if config.do_figure_show:
        plt.show()

def draw_all(config: Config, origin_data: Data, logger, predict_data_norm: np.ndarray):
    # 测试结果 画图
    _, label_data_norm = origin_data.get_all_data(return_label_data=True)
    # 通过保存的均值和方差还原数据
    label_data = label_data_norm * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]   
    predict_data = predict_data_norm * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]   
    assert label_data_norm.shape[0]==predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    label_data_d = label_data
    predict_data_d = predict_data
    mse_norm = metrics.mean_squared_error(label_data_d[config.predict_day:], predict_data_d[:-config.predict_day])
    logger.info("The MSE is " + str(mse_norm))
    rmse_norm = metrics.mean_squared_error(label_data_d[config.predict_day:], predict_data_d[:-config.predict_day])**0.5
    logger.info("The RMSE is " + str(rmse_norm))
    mae_norm = metrics.mean_absolute_error(label_data_d[config.predict_day:], predict_data_d[:-config.predict_day])
    logger.info("The MAE is " + str(mae_norm))
    r2_norm = metrics.r2_score(label_data_d[config.predict_day:], predict_data_d[:-config.predict_day])
    logger.info("The R2 is " + str(r2_norm))

    # 保存数据
    label_X = range(len(label_data))
    predict_X = [ x + config.predict_day for x in label_X]
    save_data = list(zip(label_data_d[config.predict_day:], predict_data_d[:-config.predict_day]))
    columns = ['label', 'predict']
    df = pd.DataFrame(columns=columns, data=save_data)
    df.to_csv(config.figure_save_path+"data_"+str(config.train_num)+".csv")

    #if not sys.platform.startswith('linux'):    # 无桌面的Linux下无法输出，如果是有桌面的Linux，如Ubuntu，可去掉这一行
    for i in range(label_column_num):
        plt.figure(i+1)                     # 预测数据绘制
        plt.plot(label_X, label_data[:, i], label='label')
        plt.plot(predict_X, predict_data[:, i], label='predict')
        plt.title("Predict stock {} price with {}".format(label_name[i], config.used_frame))
        logger.info("The predicted stock {} for the next {} day(s) is: ".format(label_name[i], config.predict_day) +
                str(np.squeeze(predict_data[-config.predict_day:, i])))
        if config.do_figure_save:
            plt.savefig(config.figure_save_path+"{}predict_{}_with_{}.png".format(config.continue_flag, label_name[i], config.used_frame))
    if config.do_figure_show:
        plt.show()


def main(config):
    logger = load_logger(config)
    # 设置随机种子，保证可复现
    np.random.seed(config.random_seed)  
    data_gainer = Data(config)

    if config.do_train:
        train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
        train(config, logger, [train_X, train_Y, valid_X, valid_Y], data_gainer)

    if config.do_predict:
        test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
        pred_result = predict(config, test_X)       # 这里输出的是未还原的归一化预测数据
        draw(config, data_gainer, logger, pred_result)

        '''test_X, test_Y = data_gainer.get_all_data(return_label_data=True)
        pred_result = predict(config, test_X)       # 这里输出的是未还原的归一化预测数据
        draw_all(config, data_gainer, logger, pred_result)'''
        



if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--train_data_path", default="./data2/道琼斯工业平均指数历史数据.csv", type=str, help="train_data_path")
    parser.add_argument("-n", "--train_num", default=5, type=int, help="train_num")

    args = parser.parse_args()

    con = Config()
    '''for key in dir(args):               # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):     # 去掉 args 自带属性，比如__name__等
            setattr(con, key, getattr(args, key))   # 将属性值赋给Config'''

    train_data_path = getattr(args, 'train_data_path')
    print(train_data_path)
    train_num = getattr(args, 'train_num')

    data_name = train_data_path.split("/")[-1].split(".")[0]
    log_save_path = "./result_o/"+ data_name
    figure_save_path = log_save_path+"/SVM_figure/"
    model_save_path = log_save_path+"/SVM_model/"

    log_save_path = log_save_path + "/SVM_" + data_name + '_'+ str(train_num)+ ".log"

    setattr(con, "train_data_path", train_data_path)
    setattr(con, "log_save_path", log_save_path) 
    setattr(con, "figure_save_path", figure_save_path) 
    setattr(con, "model_save_path", model_save_path)
    setattr(con, "train_num", train_num)

    make_dir(con)

    main(con)