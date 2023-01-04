import argparse
import os 
import numpy as np 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets 
from torch.autograd import Variable 
import torch.nn as nn
import torch 

os.makedirs("./images/gan/", exist_ok=True)
os.makedirs("./save/gan", exist_ok=True)
os.makedirs("./datasets/mnist", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n-epoches", type=int, default=50, help="number of epoches for training")
parser.add_argument("--batch-size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of the first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of the second order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")

opt = parser.parse_args()

print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
img_area = np.prod(img_shape) # the production of multiplication of all items in group

cuda = True if torch.cuda.is_available() else False

mnist = datasets.MNIST(
    root="./datasets/", train=True, download=True, transform=transforms.Compose(
        [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    ),
)

dataloader = DataLoader(
    mnist,
    batch_size = opt.batch_size,
    shuffle = True, 
)


## DISCRIMINATOR ##
## Flat the image 28x28=784, and through MLP, with k = 0.2 LeakyReLU
## Finally connect with sigmoid to fetch a posibility from 0 to 1
class Discriminator(nn.MOdule):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_area, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

## GENERATOR ##
## input 