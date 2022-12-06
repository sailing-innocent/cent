import numpy as np 
import matplotlib.pyplot as plt

# TODO: neural network in numpy

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

if __name__ == "__main__":
    # print(sigmoid(2))
    fig, ax = plt.subplots()
    x = np.linspace(-5.0, 5.0, 30)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.show()