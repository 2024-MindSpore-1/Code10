'''
Descripttion: The Gini-impurity preserving encryption method( transfer the GIPE_string result(string) to int and keep the order ).
version: v1
Author: anonymous
'''

import functools
import mindspore.numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


#Compare the ciphertext code1 and code2 size
def CompareTreeCode(code1, code2):
    """
        Parameters
        ----------
        code1 : string 
            The ciphertext.
        code2 : string
            The ciphertext.
    """
    if (code1 == code2):
        return 1  # code1>code2
    minLen = min(len(code1), len(code2))
    list1 = list(code1)
    list2 = list(code2)
    for i in range(minLen):
        if (list1[i] != list2[i]):
            if (list1[i] == '1'):
                return 1
            else:
                return -1
    if (minLen < len(code1)):
        if (list1[minLen] == '1'):
            return 1
        else:
            return -1
    else:
        if (list2[minLen] == '1'):
            return -1
        else:
            return 1


#Transfer the ciphertext from type string to int and kepp the order.
def str2int(data):
    int_data = []
    for col in range(len(data.iloc[0, :])):
        tmp_data = data.iloc[:, col].values.tolist()
        tmp_data = sorted(tmp_data, key=functools.cmp_to_key(CompareTreeCode))

        int_tmp = []
        for i in range(len(data.iloc[:, col])):
            index = tmp_data.index(data.iloc[i, col])
            int_tmp.append(index)
        int_data.append(int_tmp)
    int_data = pd.DataFrame(int_data)
    int_data = pd.DataFrame(int_data.values.T)
    scaler = MinMaxScaler()
    scaler = scaler.fit(int_data)
    int_data = scaler.transform(int_data)
    return int_data


if __name__ == "__main__":
    import time
    start = time.time()
    import os
    files = os.listdir("./data/OPEstr/")
    for file in files:
        dataname = file
        print(dataname)
        #training data
        data_path = './data/OPEstr/train_' + dataname
        trainingData = pd.read_csv(data_path, dtype=object)
        #testing data
        data_path = './data/OPEstr/test_' + dataname
        testingData = pd.read_csv(data_path, dtype=object)

        XTrain = trainingData.iloc[:, :-1]
        YTrain = trainingData.iloc[:, -1]
        XTest = testingData.iloc[:, :-1]
        YTest = testingData.iloc[:, -1]
        TrainLen = len(trainingData)
        TestLen = len(testingData)

        X_all = pd.concat(
            [pd.DataFrame(XTrain), pd.DataFrame(XTest)],
            axis=0,
            ignore_index=True)
        X_all.index = range(len(X_all))
        X_all = str2int(X_all)
        X_all = pd.DataFrame(X_all)

        XTrain = X_all.iloc[0:TrainLen, :]
        XTrain.index = range(len(XTrain))
        XTest = X_all.iloc[TrainLen:, :]
        XTest.index = range(len(XTest))

        pd.concat(
            [pd.DataFrame(XTrain), pd.DataFrame(YTrain)],
            axis=1).to_csv('./data/OPEint/train_' + dataname, index=0)
        pd.concat(
            [pd.DataFrame(XTest), pd.DataFrame(YTest)],
            axis=1).to_csv('./data/OPEint/test_' + dataname, index=0)
        end = time.time()
        print("time is ", end - start)
