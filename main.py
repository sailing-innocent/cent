import csv
import argparse
import numpy as np
import math
import random
from sklearn.model_selection import KFold
csvpath = "D:/workspaces/njudaily/MachineLearning/aqidataset.csv"

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

    # seperate test and train data

    folds = 10
    gap = int(len(data)/folds)
    batches = 10

    split_list = []
    kf = KFold(n_splits=folds, shuffle=True)
    # for trainIDs, testIDs in kf.split(data):
        # here trainIDs is the list of indicies that will be used for training
        # and testIDs is similiar
    
    print(i)
    print(folds, gap)



if __name__ == "__main__":
    main()

