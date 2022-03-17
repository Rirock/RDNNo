import os
import shutil 

path = './result_o_dnm30_64'
new_path = './result_o'

data_folders = os.listdir(path)
for data_folder in data_folders:
    data_path = os.path.join(path, data_folder)
    files_name = os.listdir(data_path)
    for file_name in files_name:
        if file_name.split('_')[0] == 'SVM':
            file_path = os.path.join(data_path, file_name)
            new_file_path = os.path.join(new_path, data_folder, file_name)
            if os.path.isfile(file_path):
                shutil.copy(file_path, new_file_path)
                #os.remove(file_path)
            else:
                shutil.copytree(file_path, new_file_path)