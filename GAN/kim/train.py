from pathlib import Path
from datetime import datetime
import os
import torch
import torch.optim as optim
from torchvision import transforms
import itertools

import numpy as np
import matplotlib.pyplot as plt

from model import Discriminator, Generator
from dataset import ImageDataset
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(470)
torch.cuda.manual_seed(470)


now = datetime.now()

# Hyperparameters
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_DIR, "logs", now.strftime("%Y%m%d-%H%M%S"))
LOG_ITER = 100
CKPT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCHSIZE = 4
LEARNING_RATE = 0.002
MAX_EPOCH = 200

LAMBDA = 10

if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

# Construct Data Pipeline
data_dir_X = os.path.join(Path(ROOT_DIR).parent, 'dataset', 'photo_jpg')
data_dir_Y = os.path.join(Path(ROOT_DIR).parent, 'dataset', 'monet_jpg')
transform = transforms.Compose([transforms.ToTensor()])

# Helper Functions


def imshow(img):
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def show():
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        imshow(inputs[0])


train_dataset = ImageDataset(data_dir_X, data_dir_Y, transform=transform)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=2)

Generator_XY = Generator().to(DEVICE)
Discriminator_X = Discriminator().to(DEVICE)

Generator_YX = Generator().to(DEVICE)
Discriminator_Y = Discriminator().to(DEVICE)

optimizer_generator = optim.Adam(itertools.chain(
    Generator_XY.parameters(), Generator_YX.parameters()), lr=LEARNING_RATE)
optimizer_discriminator_X = optim.Adam(
    Discriminator_X.parameters(), lr=LEARNING_RATE)
optimizer_discriminator_Y = optim.Adam(
    Discriminator_Y.parameters(), lr=LEARNING_RATE)

img_size = torch.empty(256, 256)


def train():
    writer = SummaryWriter(LOG_DIR)
    iteration = 0
    ckpt_path = os.path.join(CKPT_DIR, 'lastest.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        try:
            optimizer_discriminator_X.load_state_dict(ckpt['discriminator_X'])
            optimizer_discriminator_Y.load_state_dict(ckpt['discriminator_Y'])
            optimizer_generator.load_state_dict(ckpt['generator'])
        except RuntimeError as e:
            print('wrong checkpoint')
        else:
            print('checkpoint is loaded!')

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

            # Train G
            MSELoss = torch.nn.MSELoss().to(DEVICE)
            result_XYY = Discriminator_Y(X_to_Y).detach()
            result_YY = Discriminator_Y(input_Y).detach()
            result_YXX = Discriminator_X(Y_to_X).detach()
            result_XX = Discriminator_X(input_X).detach()
            loss_GAN_G = (MSELoss(result_XYY, torch.ones_like(result_XYY).detach(
            )) + MSELoss(result_YXX, torch.ones_like(result_YXX).detach())) / 2

            # Cycle Consistency Loss
            L1Norm = torch.nn.L1Loss().to(DEVICE)
            loss_cyc = (L1Norm(Generator_YX(Y_to_X), input_X) +
                        L1Norm(Generator_XY(Y_to_X), input_Y)) / 2 * LAMBDA

            # Identity Loss
            loss_identity = (L1Norm(X_to_Y, input_Y) +
                             L1Norm(Y_to_X, input_X)) / 2 * LAMBDA

            loss_G = loss_GAN_G + loss_cyc + loss_identity

            optimizer_generator.zero_grad()
            loss_G.backward()
            optimizer_generator.step()

            loss_DX = MSELoss(result_XX, torch.ones_like(
                result_XX).detach()) + MSELoss(result_YXX, torch.zeros_like(result_YXX.detach()))

            optimizer_discriminator_X.zero_grad()
            loss_DX.requires_grad = True
            loss_DX.backward(retain_graph=True)
            optimizer_discriminator_X.step()

            loss_DY = MSELoss(result_YY, torch.ones_like(
                result_YY).detach()) + MSELoss(result_XYY, torch.zeros_like(result_XYY).detach())
            optimizer_discriminator_Y.zero_grad()
            loss_DY.requires_grad = True
            loss_DY.backward()
            optimizer_discriminator_Y.step()

            loss = loss_G.item() + loss_DX.item() + loss_DY.item()

            if iteration % LOG_ITER == 0 and writer is not None:
                writer.add_scalar('train_loss', loss, iteration)
                print('[epoch: {}, iteration: {}] train loss : {:4f}'.format(
                    epoch+1, iteration, loss))

                ckpt = {'Discriminator_X': Discriminator_X.state_dict(),
                        'Discriminator_Y': Discriminator_Y.state_dict(),
                        'Generator_XY': Generator_XY.state_dict(),
                        'Generator_YX': Generator_YX.state_dict(),
                        'optim_discriminator_X': optimizer_discriminator_X.state_dict(),
                        'optim_discriminator_Y': optimizer_discriminator_Y.state_dict(),
                        'optim_generator': optimizer_generator.state_dict()}
                torch.save(ckpt, ckpt_path)

        print('[epoch: {}] train loss : {:4f}'.format(epoch+1, loss))


if __name__ == '__main__':
    train()
