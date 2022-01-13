import os
import glob
from PIL import Image
from random import shuffle
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, path_X, path_Y, transform = None):
        self.path_X = glob.glob(os.path.join(path_X, '*.jpg'))
        self.path_Y = glob.glob(os.path.join(path_Y, '*.jpg'))
        shuffle(self.path_X)
        shuffle(self.path_Y)
        self.transform = transform
        self.length = max(len(self.path_X), len(self.path_Y))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        img_X = Image.open(self.path_X[index % len(self.path_X)])
        img_Y = Image.open(self.path_Y[index % len(self.path_Y)])
        
        if self.transform:
            img_X = self.transform(img_X)
            img_Y = self.transform(img_Y)
        
        return img_X, img_Y