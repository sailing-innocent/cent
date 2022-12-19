import numpy as np
import csv
import math
import random
import matplotlib.pyplot as plt

def qsum(x):
    xsum = 0
    for xi in x:
        xsum = xsum + xi * xi
    return xsum

# theta: numpy.matrix() K * N
def softmax(k, x, theta):
    K = theta.shape[0]
    D = theta.shape[1]
    s = 0
    for j in range(K):
        s = s + math.exp(np.matmul(theta[j].transpose(), x))
    i = math.exp(np.matmul(theta[k].transpose(), x))
    return i / (s + 0.0001) # prevent the rare div0 error 

def sign(x, i):
    if (x == i):
        return 1
    return 0

def p(k,theta,x):
    return softmax(k, theta, x)

def gen_data(N, K):
    D = 2
    x = np.zeros([N, D],dtype=float)
    y = np.zeros(N,dtype=int)
    # generate some data range 0-1, label the data less than 0.5 by 0 and 1 the reverse
    for i in range(N):
        for j in range(D):
            x[i][j] = random.random()
        
        y[i] = K-1
        for k in range(K):
            if (k)/K*x[i][0] < x[i][1] and (k+1)/K*x[i][0] > x[i][1]:
                y[i] = k
    return x, y
    
def logistic_regression(x_samples, y_samples, K):
    N = x_samples.shape[0]
    D = x_samples.shape[1]
    theta = np.zeros([K,D], dtype=float)
    alpha = 0.01
    nsteps = 2000
    for step in range(nsteps):
        # calculate derivative
        for l in range(K):
            deri = 0
            for i in range(N):
                deri = deri + (- x_samples[i]*(sign(y_samples[i], l) - softmax(l, x_samples[i], theta)))
            theta[l] = theta[l] - alpha * deri
            # print(theta)
        if (step % 100 == 0):
            print("step: ", step, "\n deri: ", deri)

        willBreak = True
        for de in deri:
            if de > 0.03 or de < -0.03:
                willBreak = False
        if willBreak:
            break
    
    return theta, step

def logistic_regression_regu(x_samples, y_samples, K, lam = 0.1):
    N = x_samples.shape[0]
    D = x_samples.shape[1]
    theta = np.zeros([K,D], dtype=float)
    alpha = 0.005
    nsteps = 2000
    for step in range(nsteps):
        # calculate derivative
        for l in range(K):
            deri = 0
            for i in range(N):
                deri = deri + (- x_samples[i]*(sign(y_samples[i], l) - softmax(l, x_samples[i], theta)))
            theta[l] = theta[l] - alpha * ( deri + lam * theta[l])
            # print(theta)
        if (step % 100 == 0):
            print("step: ", step, "\n deri: ", deri)

        willBreak = True
        for de in deri:
            if de > 0.05 or de < -0.05:
                willBreak = False
        if willBreak:
            break
    
    return theta, step


def lreg_predict(xstar, theta):
    K = theta.shape[0]
    pmax = 0
    kmax = -1
    for i in range(K):
        pre = p(i, xstar, theta)
        if pmax < pre:
            pmax = pre
            kmax = i
    return kmax   

def test(x_test, y_test, theta):
    N = x_test.shape[0]
    res = []
    tr = 0
    for i in range(N):
        yi = predict(x[i], theta)
        if (yi == y_test[i]):
            res.append(True)
            tr = tr + 1
        else:
            res.append(False)
    # print(res)
    print("right predict rate: ", tr/N)

def debug(x, y):
    print(x)
    print(y)

    fig, ax = plt.subplots()

    ax.scatter(x[:,0], x[:,1], c=y)
    gridx = np.linspace(0.0, 1.0, 20)
    gridy = 0.5 * gridx
    ax.plot(gridx, gridy)
    plt.show()


if __name__ == "__main__":
    N = 50
    K = 2
    x, y = gen_data(N, K)
    theta = logistic_regression(x, y, K)
    test(x,y,theta)
    debug(x,y)

