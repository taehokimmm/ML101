import os
from PIL.Image import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, utils
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from model import Discriminator, Generator
from dataset import ImageDataset
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(470)
torch.cuda.manual_seed(470)

from pathlib import Path
from datetime import datetime

now = datetime.now()

# Hyperparameters
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR ="logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
LOG_ITER = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCHSIZE = 4
LEARNING_RATE = 0.002
MAX_EPOCH = 100

LAMBDA = 10


# Construct Data Pipeline
data_dir_X = os.path.join(ROOT_DIR, 'dataset', 'photo_jpg')
data_dir_Y = os.path.join(ROOT_DIR, 'dataset', 'monet_jpg')
transform = transforms.Compose([transforms.ToTensor()])

#Helper Functions
def imshow(img):
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def show():
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        imshow(inputs[0])


train_dataset = ImageDataset(data_dir_X, data_dir_Y, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=2)

Generator_XY = Generator()
Discriminator_X = Discriminator()

Generator_YX = Generator()
Discriminator_Y = Discriminator()

optimizer_generator_XY = optim.Adam(Generator_XY.parameters(), lr=LEARNING_RATE)
optimizer_discriminator_X = optim.Adam(Discriminator_X.parameters(), lr=LEARNING_RATE)
optimizer_generator_YX = optim.Adam(Generator_YX.parameters(), lr=LEARNING_RATE)
optimizer_discriminator_Y = optim.Adam(Discriminator_Y.parameters(), lr=LEARNING_RATE)

writer = SummaryWriter(LOG_DIR)

img_size = torch.empty(256, 256)

def train():
    iteration = 0
    for epoch in range(MAX_EPOCH):
        Generator_XY.train()
        Generator_YX.train()
        Discriminator_X.train()
        Discriminator_Y.train()
        for input_X, input_Y in train_dataloader:
            iteration += 1
            input_X = input_X.to(DEVICE)
            input_Y = input_Y.to(DEVICE)

            X_to_Y = Generator_XY(input_X)
            Y_to_X = Generator_YX(input_Y)

            # Adversarial Loss
            MSELoss = torch.nn.MSELoss()
            result_XYY = Discriminator_Y(X_to_Y)
            result_YY = Discriminator_Y(input_Y)
            result_YXX = Discriminator_X(Y_to_X)
            result_XX = Discriminator_X(input_X)

            loss_GAN_GX = MSELoss(result_XYY, torch.ones_like(result_XYY))
            loss_GAN_DY = MSELoss(result_YY, torch.ones_like(result_YY)) + MSELoss(result_XYY, torch.zeros_like(result_XYY))
            
            loss_GAN_GY = MSELoss(result_YXX, torch.ones_like(result_YXX))
            loss_GAN_DX = MSELoss(result_XX, torch.ones_like(result_XX)) + MSELoss(result_YXX, torch.zeros_like(result_YXX))

            loss_GAN = loss_GAN_DX + loss_GAN_DY + loss_GAN_GX + loss_GAN_GY
            # Cycle Consistency Loss
            L1Norm = torch.nn.L1Loss()
            loss_cyc = (L1Norm(Generator_YX(Y_to_X), input_X) + L1Norm(Generator_XY(Y_to_X), input_Y))*LAMBDA

            # Identity Loss
            loss_identity = (L1Norm(X_to_Y, input_Y) + L1Norm(Y_to_X, input_X))*0.5*LAMBDA


            optimizer_discriminator_X.zero_grad()
            optimizer_discriminator_Y.zero_grad()
            optimizer_generator_XY.zero_grad()
            optimizer_generator_YX.zero_grad()

            loss_GAN.backward()
            loss_cyc.backward()
            loss_identity.backward()

            optimizer_discriminator_X.step()
            optimizer_discriminator_Y.step()
            optimizer_generator_XY.step()
            optimizer_generator_YX.step()

            if iteration % 20 == 0 and writer is not None:
                writer.add_scalar('train_loss', loss.item(), iteration)
                print('[epoch: {}, iteration: {}] train loss : {:4f}'.format(epoch+1, iteration, loss.item()))
                
        print('[epoch: {}] train loss : {:4f}'.format(epoch+1, loss.item()))


if __name__ == '__main__':
    train()