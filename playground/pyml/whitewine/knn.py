# For KNN and KD-KNN
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def distance(x, y, p = 2):
    n = len(x)
    res = 0
    if (p==2):
        for i in range(n):
            res = res + (x[i] - y[i])**2
        return np.sqrt(res)

def pickSecond(elem):
    return elem[1]

def find_k_nearest(x_samples, x_test, k = 3):
    k_nearest = []
    N = len(x_samples)
    for i in range(N):
        # print("is calculating distance with: ", i, " th point: ", x_samples[i])
        dis = distance(x_samples[i], x_test)
        if i < k:
            k_nearest.append([i, dis])
            k_nearest.sort(key=pickSecond)
        else:
            if dis < k_nearest[k-1][1]:
                k_nearest[k-1] = [i, dis]
                k_nearest.sort(key=pickSecond)
        # print("current K_Near: ", k_nearest)
    return k_nearest

def knn(x_samples, y_samples, x_test, k = 3):
    k_nearest = find_k_nearest(x_samples, x_test, k)
    # print("GET K-Nearest: ", k_nearest)
    catedict = dict()
    for k_item in k_nearest:
        cate = str(int(y_samples[k_item[0]]))
        if cate in catedict.keys():
            catedict[cate] = catedict[cate] + 1
        else:
            catedict[cate] = 1
        
    maxval = 0
    res = 0
    for cate in catedict.keys():
        if catedict[cate] > maxval:
            maxval = catedict[cate]
            res = cate
    if (res == 0):
        print("SOMETHING WRONG: ", catedict)
    return res
