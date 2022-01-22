import os
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import itertools

import numpy as np
import matplotlib.pyplot as plt

from model import Discriminator, Generator
from dataset import ImageDataset
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from datetime import datetime

torch.manual_seed(470)
torch.cuda.manual_seed(470)


now = datetime.now()

# Hyperparameters
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_DIR, "logs", now.strftime("%Y%m%d-%H%M%S"))
LOG_ITER = 100
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCHSIZE = 4
LEARNING_RATE = 0.002
MAX_EPOCH = 200

LAMBDA = 10

img_size = (256, 256)

if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

# Construct Data Pipeline
data_dir_X = os.path.join(Path(ROOT_DIR).parent, "dataset", "photo_jpg")
data_dir_Y = os.path.join(Path(ROOT_DIR).parent, "dataset", "monet_jpg")
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
# Helper Functions


def initialize_weight(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0, 0.02)


train_dataset = ImageDataset(data_dir_X, data_dir_Y, transform=transform)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=2
)

Generator_XY = Generator().to(DEVICE).apply(initialize_weight)
Discriminator_X = Discriminator().to(DEVICE).apply(initialize_weight)

Generator_YX = Generator().to(DEVICE).apply(initialize_weight)
Discriminator_Y = Discriminator().to(DEVICE).apply(initialize_weight)

optimizer_generator = optim.Adam(
    itertools.chain(Generator_XY.parameters(), Generator_YX.parameters()),
    lr=LEARNING_RATE,
)
optimizer_discriminator_X = optim.Adam(Discriminator_X.parameters(), lr=LEARNING_RATE)
optimizer_discriminator_Y = optim.Adam(Discriminator_Y.parameters(), lr=LEARNING_RATE)


def schedule_lambda(epoch):
    return 1 - (epoch - MAX_EPOCH / 2 + np.abs(epoch - MAX_EPOCH / 2)) / (MAX_EPOCH)


scheduler_G = optim.lr_scheduler.LambdaLR(
    optimizer_generator, lr_lambda=schedule_lambda
)
scheduler_DX = optim.lr_scheduler.LambdaLR(
    optimizer_discriminator_X, lr_lambda=schedule_lambda
)
scheduler_DY = optim.lr_scheduler.LambdaLR(
    optimizer_discriminator_Y, lr_lambda=schedule_lambda
)

def train():
    writer = SummaryWriter(LOG_DIR)
    iteration = 0
    ckpt_path = os.path.join(CKPT_DIR, "lastest.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=torch.device(DEVICE))
        try:
            optimizer_discriminator_X.load_state_dict(ckpt["optim_discriminator_X"])
            optimizer_discriminator_Y.load_state_dict(ckpt["optim_discriminator_Y"])
            optimizer_generator.load_state_dict(ckpt["optim_generator"])
            Discriminator_X.load_state_dict(ckpt["Discriminator_X"])
            Discriminator_Y.load_state_dict(ckpt["Discriminator_Y"])
            Generator_XY.load_state_dict(ckpt["Generator_XY"])
            Generator_YX.load_state_dict(ckpt["Generator_YX"])
        except RuntimeError as e:
            print("wrong checkpoint")
        else:
            print("checkpoint is loaded!")

    for epoch in range(MAX_EPOCH):
        Generator_XY.train()
        Generator_YX.train()
        Discriminator_X.train()
        Discriminator_Y.train()
        for input_X, input_Y in train_dataloader:
            iteration += 1
            input_X = input_X.to(DEVICE)
            input_Y = input_Y.to(DEVICE)

            X_to_Y = Generator_XY(input_X).detach()
            Y_to_X = Generator_YX(input_Y).detach()

            MSELoss = torch.nn.MSELoss().to(DEVICE)
            L1Norm = torch.nn.L1Loss().to(DEVICE)

            # Train Generator
            result_XYY = Discriminator_Y(X_to_Y)
            result_YXX = Discriminator_X(Y_to_X)
            loss_GAN = (
                MSELoss(result_XYY, torch.ones_like(result_XYY))
                + MSELoss(result_YXX, torch.ones_like(result_YXX))
            ) / 2

            loss_cyc = (
                (
                    L1Norm(Generator_YX(X_to_Y), input_X)
                    + L1Norm(Generator_XY(Y_to_X), input_Y)
                )
                / 2
                * LAMBDA
            )

            loss_identity = (
                (L1Norm(X_to_Y, input_Y) + L1Norm(Y_to_X, input_X)) / 2 * LAMBDA
            )

            loss_G = loss_GAN + loss_cyc + loss_identity

            optimizer_generator.zero_grad()
            loss_G.backward()
            optimizer_generator.step()

            # Train Discriminator X
            result_XX = Discriminator_X(input_X)
            result_YXX = Discriminator_X(Y_to_X)

            loss_DX = (
                MSELoss(result_XX, torch.ones_like(result_XX))
                + MSELoss(result_YXX, torch.zeros_like(result_YXX))
            ) / 2

            optimizer_discriminator_X.zero_grad()
            loss_DX.backward()
            optimizer_discriminator_X.step()

            # Train Discriminator Y
            result_YY = Discriminator_Y(input_Y)
            result_XYY = Discriminator_Y(X_to_Y)

            loss_DY = (
                MSELoss(result_YY, torch.ones_like(result_YY))
                + MSELoss(result_XYY, torch.zeros_like(result_XYY))
            ) / 2

            optimizer_discriminator_Y.zero_grad()
            loss_DY.backward()
            optimizer_discriminator_Y.step()

            loss = loss_G.item() + loss_DX.item() + loss_DY.item()

            if iteration % LOG_ITER == 0 and writer is not None:
                writer.add_scalar("train_loss", loss, iteration)
                writer.add_scalars(
                    "generator_loss",
                    {
                        "gan_loss": loss_GAN.item(),
                        "cyc_loss": loss_cyc.item(),
                        "identity_loss": loss_identity.item(),
                    },
                    iteration,
                )
                writer.add_scalar("discriminator_loss/X", loss_DX.item(), iteration)
                writer.add_scalar("discriminator_loss/Y", loss_DY.item(), iteration)

                print(
                    "[epoch: {} iteration: {}] train loss : {:4f} GAN Loss : {:4f} Cyc Loss : {:4f} Ide Loss : {:4f}".format(
                        epoch + 1,
                        iteration,
                        loss,
                        loss_DX.item() + loss_DY.item() + loss_GAN.item(),
                        loss_cyc.item(),
                        loss_identity.item(),
                    )
                )

                ckpt = {
                    "Discriminator_X": Discriminator_X.state_dict(),
                    "Discriminator_Y": Discriminator_Y.state_dict(),
                    "Generator_XY": Generator_XY.state_dict(),
                    "Generator_YX": Generator_YX.state_dict(),
                    "optim_discriminator_X": optimizer_discriminator_X.state_dict(),
                    "optim_discriminator_Y": optimizer_discriminator_Y.state_dict(),
                    "optim_generator": optimizer_generator.state_dict(),
                }
                torch.save(ckpt, ckpt_path)

                shower = torch.cat(((input_X + 1) / 2, (X_to_Y + 1) / 2), 0)

                save_image(
                    make_grid(shower.cpu(), nrow=BATCHSIZE),
                    os.path.join(LOG_DIR, "{}.jpg".format(iteration)),
                )

        scheduler_G.step()
        scheduler_DX.step()
        scheduler_DY.step()
        print(
            "[epoch: {}] train loss : {:4f} GAN Loss : {:4f} Cyc Loss : {:4f} Ide Loss : {:4f}".format(
                epoch + 1,
                loss,
                loss_DX.item() + loss_DY.item() + loss_GAN.item(),
                loss_cyc.item(),
                loss_identity.item(),
            )
        )


if __name__ == "__main__":
    train()
