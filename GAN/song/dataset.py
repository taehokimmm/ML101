from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np


class imageDataset(Dataset):
    def __init__(self, monetDir, photoDir, randomSeed=777):
        self.monet_dir = monetDir
        self.photo_dir = photoDir
        self.data_monet = os.listdir(self.monet_dir)
        self.data_monet.sort()
        self.monet_len = len(self.data_monet)
        self.data_photo = os.listdir(self.photo_dir)
        self.data_photo.sort()
        self.photo_len = len(self.data_photo)
        #np.random.seed(randomSeed)
        #torch.manual_seed(randomSeed)
        #np.random.shuffle(self.data_photo)
        self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(p = 0.5),
                                              transforms.RandomVerticalFlip(p = 0.5),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    def __len__(self):
        return max(self.monet_len, self.photo_len)
        
    def __getitem__(self, idx):
        imageMonet = Image.open(self.monet_dir + "/" + self.data_monet[idx % self.monet_len]).convert('RGB')
        imagePhoto = Image.open(self.photo_dir + "/" + self.data_photo[idx % self.photo_len]).convert('RGB')
        imageMonet = self.transforms(imageMonet)
        imagePhoto = self.transforms(imagePhoto)
        return imageMonet, imagePhoto