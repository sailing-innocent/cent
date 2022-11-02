import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

NSamples = 20
N = 2
M = 2
theta_truth = np.zeros([N+1,M],dtype=float)
for i in range(N+1):
    for j in range(M):
        theta_truth[i][j] = i**2 + j
func = lambda x: np.matmul(theta_truth.transpose(),x)

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
    alpha = 0.0001 # learning rate
    steps = 300
    for step in range(steps):
        theta = theta - alpha *( np.matmul(sumxxt,theta) - sumxyt)
        print(theta)

    return theta
    # return theta_truth

def main():
    # create grid X
    t = np.linspace(-5,5,NSamples)
    X1,X2 = np.meshgrid(t,t)
    # X1.shape: [NSamples, NSamples]
    # construct samples according to grid
    X_samples = np.ones([NSamples*NSamples,N+1])
    for i in range(NSamples):
        for j in range(NSamples):
            X_samples[i*NSamples+j][0] = X1[i][j]
            X_samples[i*NSamples+j][1] = X2[i][j]
            X_samples[i*NSamples+j][2] = 1
    # print(X_samples[12])
    # construct y_samples with some random numbers
    y_truth = np.zeros([NSamples*NSamples,M])
    y_samples = np.zeros([NSamples*NSamples,M])

    for i in range(NSamples * NSamples):
        y_truth[i] = func(X_samples[i])
        y_samples[i] = y_truth[i] + np.random.randn(M)

    print(y_samples.shape)
    y_pred = np.zeros([NSamples,NSamples,M])
    theta_hat = lsm(X_samples,y_samples)
    for i in range(NSamples):
        for j in range(NSamples):
            y_pred[i][j] = np.matmul(theta_hat.transpose(), X_samples[i*NSamples+j])
    
    y_samples.reshape([NSamples,NSamples,M])
    print(y_samples.shape)
    y1_pred = y_pred[:,:,0]
    y1_samples = y_samples[:,0].reshape([NSamples,NSamples])
    y2_pred = y_pred[:,:,1]

    print(y1_pred.shape)
    print(y1_pred[2][0])
   
    ax = plt.axes(projection='3d')

    ax.plot_surface(X1,X2,y1_pred,cmap='viridis')
    ax.scatter3D(X1,X2,y1_samples)
    
    plt.show()


if __name__ == "__main__":
    main()
