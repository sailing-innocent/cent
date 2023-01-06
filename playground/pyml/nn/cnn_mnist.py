# according to https://zhuanlan.zhihu.com/p/112829371

import numpy as np 
import torch 
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt 
import os
from torchvision import datasets, transforms, utils 

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

if __name__ == "__main__":
    # debug()
    debug_one_batch()