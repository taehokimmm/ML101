import os
from PIL.Image import Image
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt

from model import Discriminator, Generator
from dataset import ImageDataset

# Hyperparameters
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

data_dir_X = os.path.join(ROOT_DIR, 'input')
if not os.path.exists(data_dir_X):
    os.makedirs(data_dir_X)
data_dir_Y = os.path.join(ROOT_DIR, 'output')
if not os.path.exists(data_dir_Y):
    os.makedirs(data_dir_Y)
transform = transforms.Compose([transforms.ToTensor()])
ckpt_path = os.path.join(CKPT_DIR, 'lastest.pt')

Generator_XY = Generator().to(DEVICE)
Generator_YX = Generator().to(DEVICE)

Generator_XY.eval()
Generator_YX.eval()

with torch.no_grad():
    ckpt = torch.load(ckpt_path, map_location=torch.device(DEVICE))
    Generator_XY.load_state_dict(ckpt['Generator_XY'])
    Generator_XY.load_state_dict(ckpt['Generator_XY'])

    test_dataset = ImageDataset(data_dir_X, test=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset)

    it = 0

    for inputs in test_dataloader:
        inputs = inputs.to(DEVICE)
        print(inputs.shape)

        outputs = Generator_XY(inputs)
        print(outputs.shape)

        save_image(outputs, os.path.join(data_dir_Y, "{}.png".format(it)))
        it += 1
