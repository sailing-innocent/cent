import numpy as np
from activator import *
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_dim, activator):
        self.activator = activator
        self.input_dim = input_dim
        self.weights = np.zeros(input_dim)
        self.bias = 0.0
    
    def __str__(self):
        return "current weights: \n {} \n and bias: \n {} \n".format(self.weights, self.bias)

    def forward(self, x):
        if not x.shape:
            return self.activator(self.weights * x + self.bias)
        return self.activator(np.matmul(self.weights.transpose(), x) + self.bias)

    def train(self, x_samples, y_samples, iterations, alpha):
        for i in range(iterations):
            self._one_train_iter(x_samples, y_samples, alpha)

    def _one_train_iter(self, x_samples, y_samples, alpha):
        for x_sample, y_sample in zip(x_samples, y_samples):
            y_pred = self.forward(x_sample)
            self._update_weights(x_sample, y_pred, y_sample, alpha)

    def _update_weights(self, x_input, y_pred, y_sample, alpha):
        delta = y_sample - y_pred
        self.weights = self.weights + alpha * delta * x_input
        self.bias = self.bias + alpha * delta

def test_step_perceptron():
    actFactory = ActivatorFactory()
    activator = actFactory.genActivator(1)
    x = np.ones(1)
    xdim = x.shape[0]
    p = Perceptron(xdim, activator)
    res = p.forward(x)
    print(res)
    print(p)
    x_samples = np.linspace(-1.0, 1.0, 30)
    y_samples = np.array([activator(x_sample - 0.5) for x_sample in x_samples])
    
    fig, ax = plt.subplots()
    plt.plot(x_samples, y_samples)

    print("now start training")

    iterations = 10
    p.train(x_samples, y_samples, iterations, 0.01)
    print(p)
    y_pred1 = np.array([p.forward(x_sample) for x_sample in x_samples])
    plt.plot(x_samples, y_pred1)
    plt.show()

def test_and_perceptron():
    actFactory = ActivatorFactory()
    activator = actFactory.genActivator(1)
    x = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0, 1, 1, 1])

    p = Perceptron(2, activator)
    iterations = 10
    p.train(x, y, iterations, 0.1)
    print(p)
    print('1 and 1 = %d' % p.forward(np.array([1, 1])))
    print('0 and 0 = %d' % p.forward(np.array([0, 0])))
    print('1 and 0 = %d' % p.forward(np.array([1, 0])))
    print('0 and 1 = %d' % p.forward(np.array([0, 1])))

class LineUnit(Perceptron):
    def __init__(self, input_dim):
        actFactory = ActivatorFactory()
        activator = actFactory.genActivator(0)
        Perceptron.__init__(self, input_dim, activator)

def test_reg_lineunit():
    lu = LineUnit(1)
    x_samples = np.linspace(-1.0, 1.0, 30)
    y_samples = np.array([2 * (x_sample - 0.5) for x_sample in x_samples])
    
    fig, ax = plt.subplots()
    plt.plot(x_samples, y_samples)

    print("now start training")

    iterations = 100
    lu.train(x_samples, y_samples, iterations, 0.01)
    print(lu)
    y_pred1 = np.array([lu.forward(x_sample) for x_sample in x_samples])
    plt.plot(x_samples, y_pred1)
    plt.show()

if __name__ == "__main__":
    # test_and_perceptron()
    test_reg_lineunit()