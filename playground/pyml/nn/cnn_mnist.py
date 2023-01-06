# according to https://zhuanlan.zhihu.com/p/112829371

import numpy as np 
import torch 
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt 
import os
from torchvision import datasets, transforms, utils 
import torch.nn.functional as F
import torch.optim as optim

## Prepare Data MNIST

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

train_data = datasets.MNIST(
    root = "D:/data/datasets/",
    transform = transform,
    train = True, 
    download = True
)

test_data = datasets.MNIST(
    root = "D:/data/datasets/",
    transform = transform,
    train = False
)

print("we are going to train our model on {} samples and ".format(len(train_data)))
# 60000
print("we are going to test {} samples ".format(len(test_data)))
# 10000

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=2)

print(len(test_loader)) # 10000/64 = 157

def debug():
    oneimg, label = train_data[0]
    print(oneimg.shape)
    print(label)
    oneimg = oneimg.numpy().transpose(1,2,0)
    print(oneimg.shape)
    std = [0.5]
    mean = [0.5]
    oneimg = oneimg * std + mean
    oneimg.resize(28,28)
    plt.imshow(oneimg)
    plt.show()

def debug_one_batch():
    images, labels = next(iter(train_loader))
    img = utils.make_grid(images)

    img = img.numpy().transpose(1,2,0)
    std = [0.5]
    mean = [0.5]
    img = img * std + mean
    for i in range(64): # iterate batch size
        print(labels[i], end=" ")
        i += 1
        if (i%8) == 0:
            print(end='\n')
    plt.imshow(img)
    plt.show()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def main():
    net = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_accs = []
    train_loss = []
    test_accs = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() # every 100 batch output a mean loss
            if (i%100==99):
                print("[{},{}] loss {}".format(epoch+1, i+1, running_loss/100))
                running_loss = 0.0
            train_loss.append(loss.item())
    
            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            train_accs.append(100*correct/total)
    print("Train Fin")

    PATH = "./mnist.pth"
    torch.save(net.state_dict(), PATH)

if __name__ == "__main__":
    # debug()
    # debug_one_batch()
    main()