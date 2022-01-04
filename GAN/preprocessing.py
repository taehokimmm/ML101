import os
import torch
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, utils

import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCHSIZE = 128

# Construct Data Pipeline
data_dir = os.path.join(ROOT_DIR, 'dataset')
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCHSIZE, num_workers = 2)

test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCHSIZE, num_workers = 2)

def imshow(img): 
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

def show(): 
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        imshow(inputs[0])



if __name__ == '__main__':
    show()