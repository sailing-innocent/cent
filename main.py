import csv
import argparse
import numpy as np
import math
import random
from sklearn.model_selection import KFold
csvpath = "D:/workspaces/njudaily/MachineLearning/aqidataset.csv"


def lsm(x_samples, y_samples):
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
    # gradiant update
    alpha = 0.001 # learning rate
    steps = 50
    for step in range(steps):
        theta = theta - alpha *( np.matmul(sumxxt,theta) - sumxyt)
        # print(theta)

    return theta

# Read data from CSV file and store them into numpy array
def readdata():
    data = []
    with open(csvpath, 'r', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)

    n = len(data)-1
    m = len(data[1]) - 2

    clean_data = np.zeros([n,m], dtype=float)
    # m-2 for x, 1 for y and 1 for id
    for i in range(n):
        x = [float(data[i+1][j]) for j in range(2,m)]
        x.append(float(data[i+1][1]))
        x.append(i)
        clean_data[i] = x

    # print(clean_data)

    return clean_data, n

def normalize(items):
    #L1 Normalize
    count = items.shape[0]
    n = items.shape[1]
    sumlist = np.zeros(n)

    for i in range(count):
        item = items[i,:] 
        for j in range(n):
            sumlist[j] = sumlist[j] + item[j] 

    # print(sumlist)

    for i in range(count):
        item = items[i,:]
        for j in range(n):
            item[j] = item[j]/sumlist[j] 
    
    return items

def test(x_test, y_truth, theta):
    sumres = 0
    count = x_test.shape[0]
    n = x_test.shape[1]
    m = y_truth.shape[1]

    for i in range(count):
        x = x_test[i,:].reshape(n,1)
        y = y_truth[i,:].reshape(m,1)
        res = y - np.matmul(theta.transpose(), x)
        sumres = sumres + np.matmul(res.transpose(), res)
    
    return sumres
# class model
# method train(data,ids)
# method test(data,ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", action="store_true", help="will visualize or not")
    args = parser.parse_args()
    # print(args.visualize)

    # data: [328,8+1+1]
    data, n = readdata()

    # print(data.shape)
    # test_data = data[0:2,:]
    x = data[:,0:8].reshape(n,8)
    y = data[:,8].reshape(n,1)
    x = normalize(x)
    y = normalize(y)
    # print(x.shape)
    # print(y.shape)

    # print(theta.shape)

    # seperate test and train data

    folds = 10
    gap = int(len(data)/folds)
    batches = 10

    split_list = []
    kf = KFold(n_splits=folds, shuffle=True)
    result = 0
    for trainIDs, testIDs in kf.split(data):
        #construct train and test list
        NTrain = trainIDs.shape[0]
        NTest = testIDs.shape[0]
        train_x = np.array([x[i] for i in trainIDs]).reshape(NTrain,8)
        train_y = np.array([y[i] for i in trainIDs]).reshape(NTrain,1)
        theta = lsm(train_x, train_y)
        # print("theta shape: ", theta.shape)
        test_x = np.array([x[i] for i in testIDs]).reshape(NTest,8)
        test_y = np.array([y[i] for i in testIDs]).reshape(NTest,1)
        
        res = test(test_x, test_y, theta)
        result = result + res
    print(result)

if __name__ == "__main__":
    main()

