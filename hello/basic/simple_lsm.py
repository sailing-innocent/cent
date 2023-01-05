import matplotlib.pyplot as plt
import numpy as np 

quad = lambda x: 3 * x + 1

def LSM(n, xs, ys):
    sumxy = 0
    sumx = 0
    sumy = 0
    sumxx = 0
    for i in range(n):
        sumx = sumx + xs[i]
        sumy = sumy + ys[i]
        sumxy = sumxy + xs[i] * ys[i]
        sumxx = sumxx + xs[i] * xs[i]

    k = (sumxy - n * sumy)/(sumxx - n * sumx)
    b = sumy - k * sumx
    return k,b


def gd_lsm(n, xs, ys):
    sumxy = 0
    sumx = 0
    sumy = 0
    sumxx = 0
    for i in range(n):
        sumx = sumx + xs[i]
        sumy = sumy + ys[i]
        sumxy = sumxy + xs[i] * ys[i]
        sumxx = sumxx + xs[i] * xs[i]
    alpha = 0.01
    # dL = lambda ik,ib: ik * sumxx + n * ib - sumxy, sumx * ik + n * ib - sumy
    k = 5
    b = 5
    dLdk = lambda ik,ib:sumxx*ik+sumx*ib-sumxy
    dLdb = lambda ik,ib:sumx*ik+n*ib-sumy

    for step in range(1000):
        dk = dLdk(k,b)
        db = dLdb(k,b)
        k = k - alpha * dk
        b = b - alpha * db
    return k,b

def main():
    x = np.linspace(0, 5 , 20)
    sample_y = quad(x) + np.random.randn(20)
    # k, b = LSM(20, x, sample_y)
    k, b = gd_lsm(20, x, sample_y)
    pred = lambda x: k * x + b
    pred_y = pred(x)
    print(k,b)
    fix, ax = plt.subplots()
    # ax.plot(x,y)
    ax.scatter(x, sample_y)
    ax.plot(x, pred_y)
    plt.show()

if __name__ == "__main__":
    main()