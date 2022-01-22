import os
import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import numpy as np
import matplotlib.pyplot as plt

from model import Discriminator, Generator
from dataset import ImageDataset

from pathlib import Path


# Hyperparameters
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCHSIZE = 1

data_dir_X = os.path.join(ROOT_DIR, 'input')
if not os.path.exists(data_dir_X):
    os.makedirs(data_dir_X)
data_dir_Y = os.path.join(ROOT_DIR, "output")
if not os.path.exists(data_dir_Y):
    os.makedirs(data_dir_Y)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
ckpt_path = os.path.join(CKPT_DIR, "lastest.pt")

Generator_XY = Generator().to(DEVICE)
Generator_YX = Generator().to(DEVICE)

Generator_XY.eval()
Generator_YX.eval()

with torch.no_grad():
    ckpt = torch.load(ckpt_path, map_location=torch.device(DEVICE))
    Generator_XY.load_state_dict(ckpt["Generator_XY"])
    Generator_YX.load_state_dict(ckpt["Generator_YX"])

    test_dataset = ImageDataset(data_dir_X, test=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCHSIZE)

    it = 0

    for inputs in test_dataloader:
        inputs = inputs.to(DEVICE)
        outputs = Generator_XY(inputs)

        if BATCHSIZE > 1:
            shower = torch.cat(((inputs + 1) / 2, (outputs + 1) / 2), 0)
            save_image(
                make_grid(shower.cpu(), nrow=BATCHSIZE),
                os.path.join(data_dir_Y, "{}.jpg".format(it))
            )
        else:
            save_image((outputs + 1) / 2, os.path.join(data_dir_Y, "{}.jpg".format(it)))

        it += 1
