import torch
import torch.nn as nn
from torch.nn.modules.conv import ConvTranspose2d

class ResBlock(nn.Module):
    def __init__(self, f):
        super(ResBlock, self).__init__()
        
        self.res = nn.Sequential(nn.ReflectionPad2d(1),
                                 nn.Conv2d(f, f, 3),
                                 nn.InstanceNorm2d(f),
                                 nn.ReLU(inplace=True),
                                 nn.ReflectionPad2d(1),
                                 nn.Conv2d(f, f, 3),
                                 nn.InstanceNorm2d(f))
    
    def forward(self, x):
        return x + self.res(x)
    

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        model = []
        model.extend([nn.ReflectionPad2d(3),
                      nn.Conv2d(3, 64, 7),
                      nn.InstanceNorm2d(64),
                      nn.ReLU(inplace=True)])
        for i in range(1, 3):
            model.extend([nn.ReflectionPad2d(1),
                         nn.Conv2d(64*i, 128*i, 3, stride=2),
                         nn.InstanceNorm2d(128),
                         nn.ReLU(inplace=True)])
        for i in range(9):
            model.append(ResBlock(256))
        for i in range(2, 0, -1):
            model.extend([ConvTranspose2d(128*i, 64*i, 3, stride=2, padding=1, output_padding=1),
                         nn.InstanceNorm2d(64*i),
                         nn.ReLU(inplace=True)])
        model.extend([nn.ReflectionPad2d(3),
                      nn.Conv2d(64, 3, 7),
                      nn.Tanh()])
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)


class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        
        model = []
        model.extend([nn.ReflectionPad2d(1),
                      nn.Conv2d(3, 64, 4, stride=2),
                      nn.LeakyReLU(0.2, inplace=True)])
        for i in range(3):
            model.extend([nn.ReflectionPad2d(1),
                          nn.Conv2d(64*(2**i), 128*(2**i), 4, stride=2),
                          nn.InstanceNorm2d(128*(2**i)),
                          nn.LeakyReLU(0.2, inplace=True)])
        model.extend([nn.ZeroPad2d((1, 0, 1, 0)),
                      nn.Conv2d(512, 1, 4, padding=1)])

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class ConstantandLinearlyDecay(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, start_decay, total, last_epoch=-1):
        def lr_lambda(step):
            if step < start_decay:
                return 1
            else:
                return (step - total) / (start_decay - total)
        
        super(ConstantandLinearlyDecay, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)


def initialize_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0, 0.02)
        
    
if __name__ == "__main__":
    generator = Gen()
    discriminator = Dis()
    #print(generator)
    print(discriminator)
    