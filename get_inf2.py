import os
import re
import numpy as np
from numpy.lib.index_tricks import fill_diagonal

path = "./result_o"
methods = ["DNM", "LSTM", "MLP", "ODNM", "SVM"]
patterns = ["MSE", "RMSE", "MAE", "R2"]
pattern_end = " is (\d+.?\d+)"

def open_file(path):
    try:
        with open(path, "r") as f:
            text = f.read()
    except:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    return text

def get_data_by_gp(path, folder_name):
    folder_path = os.path.join(path, folder_name)
    result_list = []
    for _ in range(len(patterns)):
        result_list.append([[] for _ in range(len(methods))])
    '''mse_list = [[] for _ in range(len(methods))]
    rmse_list = [[] for _ in range(len(methods))]
    mae_list = [[] for _ in range(len(methods))]
    r2_list = [[] for _ in range(len(methods))]
    [mse_list, rmse_list, mae_list, r2_list]'''
    # 循环10次
    for i in range(1, 11):
        for i_m in range(len(methods)):
            method = methods[i_m]
            file_name = method+"_"+folder_name+"_"+str(i)+".log"
            file_path = os.path.join(folder_path, file_name)
            text = open_file(file_path) # 读取文件

            for i_p in range(len(patterns)): # 获取信息
                pattern = "The "+ patterns[i_p] + pattern_end+'\n'
                for x in re.findall(pattern, text):
                    result_list[i_p][i_m].append(float(x))
    return result_list

def is_win(methods, result_list):
    mse_dnm = np.mean(result_list[0][0])
    for i_m in range(1, len(methods)): 
        if np.mean(result_list[0][i_m]) < mse_dnm:
            return False
    return True


if __name__ == "__main__":

    win_list = []
    folder_names = os.listdir(path)
    for folder_name in folder_names:
        result_list = get_data_by_gp(path, folder_name)
                
        # 显示信息
        print("\n"+folder_name)
        for i_m in range(len(methods)):
            print(methods[i_m], end=":  ")
            for i_p in range(len(patterns)):
                print(patterns[i_p], end=": ")
                print(np.mean(result_list[i_p][i_m]), end="\t")
            print()
        win_list.append(is_win(methods, result_list))
    
    print('win/all:({}/{})'.format(win_list.count(True), len(win_list)))
    #print(folder_names[win_list])
    from itertools import compress
    print(list(compress(folder_names, win_list)))
