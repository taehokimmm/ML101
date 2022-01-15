import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.features = self.create_layers()

    def forward(self, x):
        x = self.features(x)
        return x

    def create_layers(self):
        infos = ['c7s1-64', 'd128', 'd256', 'R256', 'R256', 'R256', 'R256',
                 'R256', 'R256', 'R256', 'R256', 'R256', 'u128', 'u64', 'c7s1-3']
        layers = []
        in_channels = 3
        for x in infos:
            if x.startswith('c7s1-'):
                layers += [nn.ReflectionPad2d(3),
                           nn.Conv2d(in_channels, int(x[5:]), kernel_size=7),
                           nn.InstanceNorm2d(int(x[5:])),
                           nn.ReLU(inplace=True)]
                in_channels = int(x[5:])

            elif x.startswith('d'):
                layers += [nn.Conv2d(in_channels, int(x[1:]), kernel_size=3, stride=2, padding=1),
                           nn.InstanceNorm2d(int(x[1:])),
                           nn.ReLU(inplace=True)]
                in_channels = int(x[1:])

            elif x.startswith('R'):
                layers += [ResidualBlock(in_channels)]

            elif x.startswith('u'):
                layers += [nn.ConvTranspose2d(in_channels, int(x[1:]), kernel_size=3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(int(x[1:])),
                           nn.ReLU(inplace=True)]
                in_channels = int(x[1:])

        return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.features = self.create_layers()

    def forward(self, x):
        return self.features(x)

    def create_layers(self):
        infos = [64, 128, 256, 512]
        layers = []
        in_channels = 3
        for x in infos:
            if x == 64:
                layers += [nn.Conv2d(in_channels, x, kernel_size=4, stride=2),
                           nn.InstanceNorm2d(x),
                           nn.LeakyReLU(0.2, inplace=True)]
                in_channels = x
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=4, stride=2),
                           nn.LeakyReLU(0.2, inplace=True)]
                in_channels = x
        layers += [nn.Conv2d(in_channels, 1, 4, padding=1)]
        return nn.Sequential(*layers)
