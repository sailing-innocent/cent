import numpy as np 
import matplotlib.pyplot as plt 
import csv 
import math
import random

from sklearn.model_selection import KFold

from logistic_reg import * 

dataset_path = "E:/data/datasets/winequality-white/winequality-white.csv"

# @name: read_dataset
# @input: 
#   - csvpath: the path to the dataset csv
#   - limit: the limitation of samples, used for test
#   - 


# @name: preprocess
# @input:
#   - data: the data to be preprocess
#   - flag: the method flag, 0 for min-max, 1 for mean, 2 for normalize
def preprocess(items):
    # print(items)
    count = items.shape[0]
    n = items.shape[1]
    sumlist = np.zeros(n)
    for i in range(count):
        item = items[i,:] 
        for j in range(n):
            sumlist[j] = sumlist[j] + item[j]

    meanlist = sumlist / count
    varlist = np.zeros(n)

    for i in range(count):
        item = items[i,:]
        for j in range(n):
            varlist[j] = varlist[j] + (item[j] - meanlist[j])**2
    for j in range(n):
        varlist[j] = math.sqrt(varlist[j]/count)

    for i in range(count):
        for j in range(n):
            items[i][j] = (items[i][j] - meanlist[j])/varlist[j]

    return items


def read_dataset(csvpath, limit = 0):
    raw_data = []
    # data = [[7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8,6][6.3,0.3,0.34,1.6,0.049,14,132,0.994,3.3,0.49,9.5,6]]
    # read the data from dataset, the data looks like, 4898 items in tatal and 12 attributes each
    ###################################################
    ## One Raw starts with row 1
    ## 0  ## fixed-acidity         ## 7
    ## 1  ## volatile acidity      ## 0.27
    ## 2  ## citric acid           ## 0.36
    ## 3  ## residual sugar        ## 20.7
    ## 4  ## chlorides             ## 0.045
    ## 5  ## free sulfur dioxide   ## 45
    ## 6  ## total sulfur dioxide  ## 170
    ## 7  ## density               ## 1.001
    ## 8  ## pH                    ## 3
    ## 9  ## sulphates             ## 0.45
    ## 10 ## alcohol               ## 8.8
    ## 11 ## quality               ## 6
    ####################################################
    with open(csvpath, 'r') as csvf:
        reader = csv.reader(csvf, delimiter=';')
        for row in reader:
            dataitem = [item for item in row]
            raw_data.append(dataitem)
        # check for data read # 324 12 [...]
        # print(len(raw_data), len(raw_data[1]), raw_data[0])
        csvf.close()
    if limit < 0:
        N = len(raw_data) - 1 # The nubmer of data items
    else:
        N = min(len(raw_data) - 1, limit)
    x_indices = [0,1,2,3,4,5,6,7,8,9,10]
    # x_indices = [0,1]
    Dx = len(x_indices)
    y_indices = [11]
    Dy = len(y_indices)


    # Construct x_data and y_data
    x_data = np.zeros(shape=[N,Dx], dtype=float)
    y_data = np.zeros(shape=[N,Dy], dtype=float)
    for i in range(1, N+1):
        for index, x_index in enumerate(x_indices):
            # remove the first line of header
            x_data[i-1][index] = float(raw_data[i][x_index])
        for index, y_index in enumerate(y_indices):
            y_data[i-1][index] = float(raw_data[i][y_index])
    # print(x_data[2,:]) # Check for data
    return x_data, y_data

def test_1(x_test, y_truth, theta):
    N = len(x_test)
    res = 0
    for i in range(N):
        x = x_test[i,:]
        y = y_truth[i,:]
        res = res + sign(y, lreg_predict(x, theta))
    
    # print("Success Rate: ", res/N)
    return res/N

def postprocess(Ns, msrs, mstbs):
    fig = plt.figure()
    ax = plt.subplot(121)
    ax.scatter(Ns, msrs)
    ax = plt.subplot(122)
    ax.scatter(Ns, mstbs)
    plt.show()

def pickSecond(elem):
    return elem[1]

def get_most_important_factor(theta):
    theta = list(theta.transpose())
    M = 3 # most influencial 3 factors
    maxheap = []
    for i, val in enumerate(theta):
        dis = qsum(val)
        if (i < M):
            maxheap.append([i, dis])
            maxheap.sort(key=pickSecond)
        else:
            if dis > maxheap[0][1]:
                maxheap[0] = [i, dis]
                maxheap.sort(key=pickSecond)
    return maxheap

def experiment_1():
    N0 = 100
    NStride = 4000
    msrs = []
    mstbs = []
    Ns = []

    for i in range(2):
        N = N0 + i * NStride
        print("IS Training with ", N, " Samples")
        Ns.append(N)
        x, y = read_dataset(dataset_path, N)
        x = preprocess(x)
        # debug_data(x,y)
        # check_labels(y) # 3,4,5,6,7,8,9

        K = 10
        n_folds = 5
        kf = KFold(n_splits=n_folds, shuffle=True)
        Dx = x.shape[1]
        Dy = y.shape[1]
        N = x.shape[0]

        msr = 0
        mstb = 0

        for trainIDs, testIDs in kf.split(x):
            NTrain = trainIDs.shape[0]
            NTest = testIDs.shape[0]
            x_train = np.array([x[i,:] for i in trainIDs]).reshape(NTrain, Dx)
            y_train = np.array([y[i,:] for i in trainIDs]).reshape(NTrain, Dy)
            x_test = np.array([x[i] for i in trainIDs]) #.reshape(NTest, Dx)
            y_test = np.array([y[i] for i in trainIDs]) #.reshape(NTest, Dy)
            # predicted theta and loss
            theta, step = logistic_regression(x_train, y_train, K)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("current theta: ", theta)
            sr = test_1(x_test, y_test, theta)
            msr = msr + sr
            mstb = mstb + step


        msr = msr/n_folds
        mstb = mstb/n_folds
        print("success rate: ", msr)
        print("mean converge steps: ", mstb)
        msrs.append(msr)
        mstbs.append(mstb)
    
    most_important_factors = get_most_important_factor(theta)
    print(most_important_factors)
    postprocess(Ns, msrs, mstbs)
    


def check_labels(y):
    labels = dict()
    for iy in y:
        labels[str(iy[0])] = 1
    print(labels)    

def debug_data(x, y):
    fix, ax = plt.subplots()
    colors = []
    for iy in y:
        colors.append([0.2, iy/10, 0.2])
    ax.scatter(x[:,9], x[:,10], c=colors)
    plt.show()

if __name__ == "__main__":
    experiment_1()