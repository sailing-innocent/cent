import numpy as np
import csv
import math
import random
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# the file path to the dataset csv file
dataset_path = "E:/data/datasets/aqi/aqidataset.csv"

# @name: read_dataset
# @input: the path to the dataset csv
# @output: numpy array data_x and data_y
def read_dataset(csvpath):
    raw_data = []
    # read the data from dataset, the data looks like
    ###################################################
    ## One Raw starts with row 1
    ## 0  ## City Name             ## Ngawa Prefecture
    ## 1  ## AQI                   ## 23
    ## 2  ## Precipitation         ## 665.1
    ## 3  ## GDP                   ## 271.13
    ## 4  ## Temperature           ## 8.2 
    ## 5  ## Longitude             ## 102.22465
    ## 6  ## Latitude              ## 31.89941
    ## 7  ## Altitude              ## 2617
    ## 8  ## Population Density    ## 11
    ## 9  ## Coastal               ## 0
    ## 10 ## Green CoverageRate    ## 36
    ## 11 ## Incineration(10000ton)## 23
    ####################################################

    with open(csvpath, 'r') as csvf:
        reader = csv.reader(csvf)
        for row in reader:
            dataitem = [item for item in row]
            raw_data.append(dataitem)
        # check for data read # 324 12 [...]
        # print(len(raw_data), len(raw_data[1]), raw_data[1])
        csvf.close()
    
    N = len(raw_data) - 1 # The nubmer of data items
    # choose 
    # * Precipitation(2)
    # * GDP(3)
    # * Temperature(4) 
    # * Longitude(5)
    # * Latitude(6)
    # * Altitude(7)
    # * Population Density (8)
    # * Coastal (9)
    # * Green Coverage Rate (10)
    # * Incrineration (11)
    # as input 
    x_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    Dx = len(x_indices) # The dimension of x
    # choose AQI(1) as output
    y_indices = [1]
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

# @name: preprocess
# @input:
#   - data: the data to be preprocess
#   - flag: the method flag, 0 for min-max, 1 for mean, 2 for normalize
def preprocess(items, flag):
    # print(items)
    count = items.shape[0]
    n = items.shape[1]
    sumlist = np.zeros(n)
    maxlist = np.zeros(n)
    minlist = np.ones(n) * 10000

    for i in range(count):
        item = items[i,:] 
        for j in range(n):
            sumlist[j] = sumlist[j] + item[j]
            if (maxlist[j] < item[j]):
                maxlist[j] = item[j]
            if (minlist[j] > item[j]):
                minlist[j] = item[j]
    print("min:", minlist)
    print("max:", maxlist)
    meanlist = sumlist / count
    print('mean: ', meanlist)
    varlist = np.zeros(n)

    for i in range(count):
        item = items[i,:]
        for j in range(n):
            varlist[j] = varlist[j] + (item[j] - meanlist[j])**2
    for j in range(n):
        varlist[j] = math.sqrt(varlist[j]/count)
    print('var: ', varlist)

    if (flag == 0):
        # Min-Max
        for i in range(count):
            for j in range(n):
                items[i][j] = (items[i][j] - minlist[j])/(maxlist[j]-minlist[j])
        print("preproces min_max: ", items[0])
    elif (flag == 1):
        # Mean
        for i in range(count):
            for j in range(n):
                items[i][j] = (items[i][j] - meanlist[j])/(maxlist[j]-minlist[j])
        print("preprocess mean: ", items[0])
    elif (flag == 2):
        # Standarlization
        for i in range(count):
            for j in range(n):
                items[i][j] = (items[i][j] - meanlist[j])/varlist[j]
        pass
    else:
        pass
    return items


def ridge(x_samples, y_samples, alpha = 0.0001, lam = 0.1):
    count = x_samples.shape[0]
    n = x_samples.shape[1]
    m = y_samples.shape[1]

    #init theta 
    theta = np.zeros([n,m], dtype=float)
    # build params
    sumxxt = 0
    sumxyt = np.zeros([n,m], dtype=float)

    for i in range(count):
        x = x_samples[i,:].reshape([n,1])
        y = y_samples[i,:].reshape([m,1])
        sumxxt = sumxxt + np.matmul(x,x.transpose())
        sumxyt = sumxyt + np.matmul(x,y.transpose())

    gradient = np.matmul(sumxxt,theta) - sumxyt + lam * signal(theta) * theta
    # gradiant update
    steps = 5000000 # max steps
    # print(sumxxt)
    loss = []

    ERR = 0.01
    step = 0

    while (np.linalg.norm(x=gradient) > ERR and step < steps):
        gradient = np.matmul(sumxxt,theta) - sumxyt
        theta = theta - alpha * gradient
        step = step + 1

    return theta, step


def lsm(x_samples, y_samples, alpha = 0.001):
    count = x_samples.shape[0]
    n = x_samples.shape[1]
    m = y_samples.shape[1]

    #init theta 
    theta = np.zeros([n,m], dtype=float)
    # build params
    sumxxt = 0
    sumxyt = np.zeros([n,m], dtype=float)

    for i in range(count):
        x = x_samples[i,:].reshape([n,1])
        y = y_samples[i,:].reshape([m,1])
        sumxxt = sumxxt + np.matmul(x,x.transpose())
        sumxyt = sumxyt + np.matmul(x,y.transpose())
    gradient = np.matmul(sumxxt,theta) - sumxyt
    # gradiant update
    steps = 5000000 # max steps
    # print(sumxxt)
    loss = []

    ERR = 0.01
    step = 0

    while (np.linalg.norm(x=gradient) > ERR and step < steps):
        gradient = np.matmul(sumxxt,theta) - sumxyt
        theta = theta - alpha * gradient
        step = step + 1

    return theta, step

def lwlr(x_samples, y_samples, x_test, y_test, k = 1):
    n = x_samples.shape[0]
    m = y_samples.shape[0]
    tn = x_test.shape[0]
    weights = eye(n)
    loss = 0
    for itest in range(tn):
        for isample in range(n):
            diff = x_test[itest,:] - x_samples[isample,:]
            weights[isample, isample] = np.exp(np.matmul(diff, diff.transpose())/(-2.0 * k ** 2))
        xTx = np.matmul(np.matmul(x_samples.transpose(), weights),x_samples)
        if np.linalg.det(xTx) == 0.0:
            print("singular")
            return
        theta = xTx.I * (x_samples.T * (weights * y_samples.T))



def signal(theta):
    n = theta.shape[0]
    m = theta.shape[1]
    res = np.zeros([n,m], dtype=float)
    for i in range(n):
        for j in range(m):
            if (theta[i][j] > 0):
                res[i][j] = 1
            elif (theta[i][j] < 0):
                res[i][j] = -1

    return res



def lasso(x_samples, y_samples, alpha = 0.0001, lam = 0.1):
    count = x_samples.shape[0]
    n = x_samples.shape[1]
    m = y_samples.shape[1]

    #init theta 
    theta = np.zeros([n,m], dtype=float)
    # build params
    sumxxt = 0
    sumxyt = np.zeros([n,m], dtype=float)

    for i in range(count):
        x = x_samples[i,:].reshape([n,1])
        y = y_samples[i,:].reshape([m,1])
        sumxxt = sumxxt + np.matmul(x,x.transpose())
        sumxyt = sumxyt + np.matmul(x,y.transpose())

    gradient = np.matmul(sumxxt,theta) - sumxyt + lam * signal(theta)
    # gradiant update
    steps = 5000000 # max steps
    # print(sumxxt)
    loss = []

    ERR = 0.01
    step = 0

    while (np.linalg.norm(x=gradient) > ERR and step < steps):
        gradient = np.matmul(sumxxt,theta) - sumxyt
        theta = theta - alpha * gradient
        step = step + 1

    return theta, step



def test(x_test, y_truth, theta):
    sumres = 0
    count = x_test.shape[0]

    for i in range(count):
        x = x_test[i,:]
        y = y_truth[i,:]
        res = y - np.matmul(theta.transpose(), x)
        sumres = sumres + np.matmul(res.transpose(), res)
    
    return sumres.reshape(1)[0]

def experiment_lsm(x, y, alphas):
    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True)
    Dx = x.shape[1]
    Dy = y.shape[1]
    N = x.shape[0]

    msteps = []
    loss = []
    for alpha in alphas:
        meanstep = 0
        _loss = 0
        for trainIDs, testIDs in kf.split(x):
            NTrain = trainIDs.shape[0]
            NTest = testIDs.shape[0]
            x_train = np.array([x[i,:] for i in trainIDs]).reshape(NTrain, Dx)
            y_train = np.array([y[i,:] for i in trainIDs]).reshape(NTrain, Dy)
            x_test = np.array([x[i] for i in trainIDs]) #.reshape(NTest, Dx)
            y_test = np.array([y[i] for i in trainIDs]) #.reshape(NTest, Dy)
            # predicted theta and loss
            theta, step = lsm(x_train, y_train, alpha)
            # theta, step = lasso(x_train, y_train, x_test, y_test)
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            meanstep = step + meanstep

            _loss = _loss + test(x_test, y_test, theta)
        meanstep = meanstep / n_folds
        _loss = _loss / n_folds
        msteps.append(meanstep)
        loss.append(_loss)
    print(theta)
    return msteps, loss

def experiment_lasso(x, y, alphas):
    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True)
    Dx = x.shape[1]
    Dy = y.shape[1]
    N = x.shape[0]

    msteps = []
    loss = []

    meanstep = 0
    _loss = 0
    for lam in lams:
        for trainIDs, testIDs in kf.split(x):
            NTrain = trainIDs.shape[0]
            NTest = testIDs.shape[0]
            x_train = np.array([x[i,:] for i in trainIDs]).reshape(NTrain, Dx)
            y_train = np.array([y[i,:] for i in trainIDs]).reshape(NTrain, Dy)
            x_test = np.array([x[i] for i in trainIDs]) #.reshape(NTest, Dx)
            y_test = np.array([y[i] for i in trainIDs]) #.reshape(NTest, Dy)
            # predicted theta and loss
            theta, step = lasso(x_train, y_train, lam)
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            meanstep = step + meanstep

            _loss = _loss + test(x_test, y_test, theta)
        meanstep = meanstep / n_folds
        _loss = _loss / n_folds
        msteps.append(meanstep)
        loss.append(_loss)
    print(theta)
    return msteps, loss



def experiment_ridge(x, y, lams):
    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True)
    Dx = x.shape[1]
    Dy = y.shape[1]
    N = x.shape[0]

    msteps = []
    loss = []

    meanstep = 0
    _loss = 0
    for lam in lams:
        for trainIDs, testIDs in kf.split(x):
            NTrain = trainIDs.shape[0]
            NTest = testIDs.shape[0]
            x_train = np.array([x[i,:] for i in trainIDs]).reshape(NTrain, Dx)
            y_train = np.array([y[i,:] for i in trainIDs]).reshape(NTrain, Dy)
            x_test = np.array([x[i] for i in trainIDs]) #.reshape(NTest, Dx)
            y_test = np.array([y[i] for i in trainIDs]) #.reshape(NTest, Dy)
            # predicted theta and loss
            theta, step = ridge(x_train, y_train, lam)
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            meanstep = step + meanstep

            _loss = _loss + test(x_test, y_test, theta)
        meanstep = meanstep / n_folds
        _loss = _loss / n_folds
        msteps.append(meanstep)
        loss.append(_loss)
    print(theta)
    return msteps, loss

def experiment_lswr(x, y, ks):
    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True)
    Dx = x.shape[1]
    Dy = y.shape[1]
    N = x.shape[0]
    loss = []
    _loss = 0
    for k in ks:
        for trainIDs, testIDs in kf.split(x):
            NTrain = trainIDs.shape[0]
            NTest = testIDs.shape[0]
            x_train = np.array([x[i,:] for i in trainIDs]).reshape(NTrain, Dx)
            y_train = np.array([y[i,:] for i in trainIDs]).reshape(NTrain, Dy)
            x_test = np.array([x[i] for i in trainIDs]) #.reshape(NTest, Dx)
            y_test = np.array([y[i] for i in trainIDs]) #.reshape(NTest, Dy)
            _loss = _loss + lwlr(x_train, y_train, x_test, y_test, k)

        _loss = _loss / n_folds
        loss.append(_loss)
    print(theta)
    return loss

################
# MAIN
################
if __name__ == "__main__":
    x_data, y_data = read_dataset(dataset_path)
    # print(x_data.shape, y_data.shape) # check the shape as expected

    # different preprocess for input data
    flag = 2 # 1 for min-max, 2 for mean, 3 for standarlize
    
    fig, ax = plt.subplots()

    x = preprocess(x_data, flag)
    y = preprocess(y_data, flag) 
    # experiment
    # labels = ['min-max', 'mean', 'standarlize']

    # alphas = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    # lams = [0.002, 0.001, 0.0005, 0.0001]
    ks = [1, 0.1, 0.01]
    # msteps, loss = experiment_ridge(x,y, lams)
    # msteps, loss = experiment_lasso(x,y, lams)
    # msteps, loss = experiment_lsm(x, y, alphas)
    loss = experiment_lswr(x, y, ks)
    # ax.scatter(alphas, msteps, label=labels[flag])
    # ax.scatter(alphas, loss, label='loss')
    
    # ax.scatter(lams, msteps, label='msteps')
    # ax.scatter(lams, loss, label='loss')
    ax.scatter(ks, loss, label='loss')
    ax.set_xscale("log")
    plt.xlabel('lambda')
    # plt.xlabel('alpha')
    plt.ylabel('coverage-step')
    plt.legend()
    plt.show()

