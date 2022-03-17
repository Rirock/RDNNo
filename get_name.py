import os
path = "./data3/"
num = 10
files = os.listdir(path)
for file in files:
    filePath = path+file
    for i in range(1,num+1):
        #print("python MLP_main.py -d "+'"'+filePath+'" -n '+str(i))
        #print("python print_plot.py -d "+'"'+filePath+'" -n '+str(i)+' -m "DNM"')
        print("python SVM_main.py -d "+'"'+filePath+'" -n '+str(i))