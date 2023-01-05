import torch
import datetime
import numpy as np
import torch.nn as nn 
import torch.optim as optim
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import axes3d

"""Use Chinese
mpl.rcParams["font.sans-serif"] = ['FangSong']
mpl.rcParams["axes.unicode_minus"] = False 
"""

num_time_steps = 16 # step time for training time
hidden_size = 16 # we try to predict 16th point given 15th samples
input_size = 3 # 3 dim point
output_size = 3 # predict the next position
num_layers = 1
lr = 0.01

class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(Net, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        for p in self.rnn.parameters():
          nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):

       out, hidden_prev = self.rnn(x, hidden_prev)
       # [b, seq, h]
       out = out.view(-1, hidden_size)
       out = self.linear(out)#[seq,h] => [seq,3]
       out = out.unsqueeze(dim=0)  # => [1,seq,3]
       return out, hidden_prev

def train_RNN(data):

    model = Net(input_size, hidden_size, num_layers)
    print('model:\n',model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    #初始化h
    hidden_prev = torch.zeros(1, 1, hidden_size)
    l = []
    # 训练3000次
    for iter in range(3000):
        # loss = 0
        start = np.random.randint(10, size=1)[0]
        end = start + 15
        x = torch.tensor(data[start:end]).float().view(1, num_time_steps - 1, 3)
        # 在data里面随机选择15个点作为输入，预测第16
        y = torch.tensor(data[start + 5:end + 5]).float().view(1, num_time_steps - 1, 3)

        output, hidden_prev = model(x, hidden_prev)
        hidden_prev = hidden_prev.detach()

        loss = criterion(output, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print("Iteration: {} loss {}".format(iter, loss.item()))
            l.append(loss.item())


    ##############################绘制损失函数#################################
    plt.plot(l,'r')
    plt.xlabel('train times')
    plt.ylabel('loss')
    plt.title('RNN Loss Function Descend')

    return hidden_prev,model
#############################预测#########################################

def RNN_pre(model, data, hidden_prev):
    data_test = data[19:29]
    data_test = torch.tensor(np.expand_dims(data_test, axis=0),dtype=torch.float32)

    pred1,h1 = model(data_test,hidden_prev )
    print('pred1.shape:',pred1.shape)
    pred2,h2 = model(pred1,hidden_prev )
    print('pred2.shape:',pred2.shape)
    pred1 = pred1.detach().numpy().reshape(10,3)
    pred2 = pred2.detach().numpy().reshape(10,3)
    predictions = np.concatenate((pred1,pred2),axis=0)
    # predictions= mm.inverse_transform(predictions)
    print('predictions.shape:',predictions.shape)
    ## DEBUG
    ax = plt.figure().add_subplot(projection='3d')
    # ax.scatter3D(data[:,0], data[:,1], data[:,2], c="red")
    ax.scatter3D(predictions[:,0], predictions[:,1], predictions[:,2], c="y")
    ax.set_xlabel('X')
    ax.set_xlim(0, 8.5)
    ax.set_ylabel('Y')
    ax.set_ylim(0, 10)
    ax.set_zlabel('Z')
    ax.set_zlim(0, 4)
    plt.title("init data")
    plt.show()
## init data
def getdata():
    x1 = np.linspace(1,10,30).reshape(30, 1)
    y1 = (np.zeros_like(x1)+2)+np.random.rand(30,1)*0.1
    z1 = (np.zeros_like(x1)+2).reshape(30,1)
    tr1 = np.concatenate((x1,y1,z1),axis=1)
    return tr1

def debug(data):
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c="red")
    ax.set_xlabel('X')
    ax.set_xlim(0, 8.5)
    ax.set_ylabel('Y')
    ax.set_ylim(0, 10)
    ax.set_zlabel('Z')
    ax.set_zlim(0, 4)
    plt.title("init data")
    plt.show()

def main():
    data = getdata()
    # debug(data)
    # calc time
    hidden_pre, model = train_RNN(data)
    RNN_pre(model, data, hidden_pre)
    

if __name__=="__main__":
    main()