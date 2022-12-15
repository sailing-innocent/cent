# For KNN and KD-KNN
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def distance(x, y, p = 2):
    n = x.shape[0]
    res = 0

    if (p==2):
        for i in range(n):
            res = res + (x[i] - y[i])**2
        return np.sqrt(res)

def find_k_nearest(x_samples, x_test, k = 3):
    pass


def knn(x_samples, y_samples, x_test, k = 3):
    N = x_samples.shape[0]
    catedict = dict()
    for i in range(k):
        min_dis = distance(x_test, x_samples[i])

        for j in range(i+1, N):
            d = distance(x_test, x_samples[j]) 
            if d < min_dis:
                min_dis = d
                # print(" is swaping: ", i, " ", j)
                x_samples[[i,j],:] = x_samples[[j,i], :]
                y_samples[[i,j]] = y_samples[[j,i]]
        
        cate = str(y_samples[i])
        catedict[cate] = 0

    for i in range(k):
        cate = str(y_samples[i])
        catedict[cate] = catedict[cate]  + 1
    
    maxval = 0
    res = y_samples[0]
    for cate in catedict.keys():
        if catedict[cate] > maxval:
            maxval = catedict[cate]
            res = cate

    # print(res)
    # print(x_samples[0], x_samples[1])
    return res

