import numpy as np

def identityActivator(x):
    return x

def stepActivator(x):
    if x > 0:
        return 1
    return 0

class ActivatorFactory:
    def __init__(self):
        pass

    def genActivator(self, id):
        if (id == 0):
            return identityActivator
        if (id == 1):
            return stepActivator

