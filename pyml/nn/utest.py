from mynn import *

def show_sigmoid_func():
    x = np.linspace(-10,10,30)
    y = sigmoid(x)
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.savefig("D:/data/figures/sigmoid_func.png")

def show_simple_neuron():
    weights = np.array([0, 1])
    bias = 4
    n = Neuron(weights, bias)
    x = np.array([2,3])
    print(n.feedforward(x)) # 0.999
    print(sigmoid(0*2+1*3+4)) # 0.999

def show_simple_network():
    network = NeuralNetwork()
    x = np.array([2,3])
    print(network.feedforward(x)) # 0.721

def show_mse_loss():
    y_true = np.array([0,1,1,0])
    y_pred = np.array([0,0,0,0])
    print(mse_error(y_true, y_pred)) # 0.5


def preprocess(x):
    N = len(x)
    if (N <= 0):
        return x
    else:
        NDim = len(x[0])

    x_sum = np.zeros([NDim], dtype=float)

    for i in range(N):
        for j in range(NDim):
            x_sum[j] = x_sum[j] + x[i][j]

    x_mean = x_sum / N
    x_var = np.zeros([NDim], dtype=float)
    for i in range(N):
        for j in range(NDim):
            x_var[j] = x_var[j] + (x[i][j] - x_mean[j]) ** 2
    for j in range(NDim):
        x_var[j] = np.sqrt(x_var[j]/N)

    for i in range(N):
        for j in range(NDim):
            x[i][j] = (x[i][j] - x_mean[j])/x_var[j]
    return x
        

def train_simple_network():
    """
    |Alice | 133 | 65 | F |
    |Bob | 160 | 72 | M |
    |Charlie | 152 | 70 | M |
    |Diana | 120 | 60 | F |
    """
    data_x = np.array([[133,65],[160,72],[152,70],[120,60]], dtype=float)
    data_y = np.array([0,1,1,0])
    prep_x = preprocess(data_x)
    # print(prep_x)
    network = NeuralNetwork()
    network.train(prep_x, data_y)
    """
    |Emma | 135 | 68 | F |
    """
    test_x = np.array([0.8, 0.7])
    pred_y = network.feedforward(test_x)
    print(pred_y)
    # make prediction

if __name__ == "__main__":
    # show_sigmoid_func()
    # show_simple_neuron()
    # show_simple_network()
    train_simple_network()
    # show_mse_loss()