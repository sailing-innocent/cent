import numpy as np 
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx*(1-fx)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights 
        self.bias = bias

    def feedforward(self, x):
        return sigmoid(np.dot(self.weights, x) + self.bias)

class NeuralNetwork:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        
        weights = np.array([0,1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        out_h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        out_o1 = sigmoid(self.w5 * out_h1 + self.w6 * out_h2 + self.b3)
        return out_o1
    
    def train(self, all_x, all_y_true):
        learn_rate = 0.1
        epoches = 1000
        for epoch in range(epoches):
            for x, y_true in zip(all_x, all_y_true):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)

                y_pred = o1

                d_L_d_ypred = -2 * (y_true - y_pred)
                d_ypred_d_sumo1 = deriv_sigmoid(sum_o1)
                d_ypred_d_w5 = h1 * d_ypred_d_sumo1
                d_ypred_d_w6 = h2 * d_ypred_d_sumo1
                d_ypred_d_b3 = d_ypred_d_sumo1

                d_ypred_d_h1 = self.w5 * d_ypred_d_sumo1
                d_ypred_d_h2 = self.w6 * d_ypred_d_sumo1

                # Neuron h1

                d_h1_d_sumh1 = deriv_sigmoid(sum_h1)
                d_h1_d_w1 = x[0] * d_h1_d_sumh1
                d_h1_d_w2 = x[1] * d_h1_d_sumh1
                d_h1_d_b1 = d_h1_d_sumh1

                # Neuron h2
                d_h2_d_sumh2 = deriv_sigmoid(sum_h2)
                d_h2_d_w3 = x[0] * d_h2_d_sumh2
                d_h2_d_w4 = x[1] * d_h2_d_sumh2
                d_h2_d_b2 = d_h2_d_sumh2

                ## ------------------ UPDATE ------------------------------

                # Neuron h1
                self.w1 = self.w1 - learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 = self.w2 - learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 = self.b1 - learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 = self.w3 - learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 = self.w4 - learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.b2 = self.b2 - learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 = self.w5 - learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w4 = self.w6 - learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b2 = self.b3 - learn_rate * d_L_d_ypred * d_ypred_d_b3

        
            # --- Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, all_x)
                loss = mse_loss(all_y_true, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

