'''
Descripttion: Privacy preserving random forests(decision tree).
version: v2
Author: anonymous
'''
from sklearn import tree
import mindspore.numpy as np
import pickle
import time
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import os

files = os.listdir("./data/OPEminmax_cipher/")
acc = []
std = []
for file in files:
    dataName = file
    print(dataName)
    trainingData = pd.read_csv('./data/OPEminmax_cipher/' + dataName)
    print(trainingData)
    X_train, X_test, Y_train, Y_test = train_test_split(
        trainingData.iloc[:, :-1],
        trainingData.iloc[:, -1],
        test_size=0.1,
        shuffle=True,
        random_state=100)
    trainingData = pd.concat(
        [pd.DataFrame(X_train), pd.DataFrame(Y_train)], axis=1)
    testData = pd.concat([pd.DataFrame(X_test), pd.DataFrame(Y_test)], axis=1)
    # clf = RandomForestClassifier(max_features="auto",bootstrap=True,n_estimators=100,max_leaf_nodes=10)
    clf = DecisionTreeClassifier()
    start = time.time()
    clf = clf.fit(np.array(trainingData.iloc[:, :-1]),
                  np.array(trainingData.iloc[:, -1]))
    end = time.time()
    print("training time is:", end - start)

    start = time.time()
    score = clf.score(np.array(testData.iloc[:, :-1]),
                      np.array(testData.iloc[:, -1]))
    end = time.time()
    print("testing time is:", end - start)
    print(score)
# # Parameter adjustment
parameters = {'n_estimators': range(30,100,10),'max_leaf_nodes':range(10,100,10)}
grid_rfc = GridSearchCV(clf,parameters,scoring='f1_macro')
grid_rfc.fit(trainingData.iloc[:,:-1],trainingData.iloc[:,-1])
print(grid_rfc.best_params_,grid_rfc.best_score_)

