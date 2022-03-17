import os
import shutil 

path = './result_o'
method = 'DNM'

data_folders = os.listdir(path)
max_num = len(data_folders)
num = 0
for data_folder in data_folders:
    data_path = os.path.join(path, data_folder)
    files_name = os.listdir(data_path)
    for file_name in files_name:
        if file_name.split('_')[0] == method:
            num += 1
            break

print("当前进度：（{}/{}）".format(num, max_num))