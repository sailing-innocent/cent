import numpy as np

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

def dsigmoiddt(x):
    return sigmoid(x)*(1- sigmoid(x))

def tanh(x):
    return np.tanh(x)

def dtanhdt(x):
    t = tanh(x)
    return 1 - t * t

def identityActivator(x):
    return x

def stepActivator(x):
    if x > 0:
        return 1
    return 0

def sigmoidActivator(x):
    return sigmoid(x)

def tanhActivator(x):
    return tanh(x)

class ActivatorFactory:
    def __init__(self):
        pass

    def genActivator(self, id):
        if (id == 0):
            return identityActivator
        if (id == 1):
            return stepActivator

