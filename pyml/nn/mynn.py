import numpy as np 
from activator import * 
import matplotlib.pyplot as plt

# neural network is definitely a perceptron, but it changes the step function to sigmoid function

class Neuron:
    def __init__(self, layerid = 0, nodeid = 0, input_dim = 0):
        self.layerid = layerid
        self.nodeid = nodeid
        self.weight = np.zeros(input_dim, dtype=float)
        self.bias = 0
        self.output = 0
        self.delta = 0

    def __str__(self):
        return "the node {},{} has weight: {} and bias {}".format(self.layerid, self.nodeid, self.weight, self.bias)

    def set_output(self, output):
        self.output = output

class Layer:
    def __init__(self, layerid = 0, input_dim = 0, n_nodes = 0):
        self.layerid = layerid
        self.n_nodes = n_nodes
        self.nodes = []

    def init_nodes(self):
        pass
    
    def set_output(self, output):
        for i in range(n_nodes):
            self.nodes[i].set_output(output[i])

    def forward(self, x):
        pass 

    def backprop()

class Network:
    def __init__(self, struct=[1,1]):
        self.input_dim = struct[0]
        self.n_layers = len(struct)
        self.output_dim = struct[nlayers-1]
        self.layers = []
        # create input layer
        for i in range(1, self.n_layers-1):
            # create hidden layer
            pass

        # create output layer
    
    def create_input_layer(self):
        pass

    def create_output_layer(self):
        pass
    
    def create_hidden_layer(self, layerid):
        pass
    
    def __str__(self):
        return "this network gets intput {} dim vector and output {} dim vector".format(self.input_dim, self.output_dim)

    def forward(self, x):
        # input
        input_layer = layers[0]
        input_layer.set_output(x)
        for i in range(1, self.n_layers):
            prev = layers[i-1].output()
            layers[i].forward(prev)
        # get the output from the output of layers[self.n_layers-1].output
        return self.layers[self.n_layers-1].output()

    def train(self, x_trains, y_trains, alpha = 0.01):
        # train for samples 
        N = x_train.shape[0]
        for i in range(N):
            y_pred = self.forward(x_trains[i])
            self.backprop(y_pred, y_trains[i])

    def backprop(self):
        pass
    
    def test(self, x_tests, y_tests):
        # test the 
        pass

    def debug(self):
        pass



if __name__ == "__main__":
    # print(sigmoid(2))


    fig, ax = plt.subplots()
    x = np.linspace(-5.0, 5.0, 30)


    plt.plot(x, y)
    plt.show()