import os
import glob
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, path_X, path_Y=None, transform=None, test=False):
        self.path_X = glob.glob(os.path.join(path_X, "*.jpg"))
        self.test = test
        if test == False:
            self.path_Y = glob.glob(os.path.join(path_Y, "*.jpg"))
            self.length = max(len(self.path_X), len(self.path_Y))
        else:
            self.length = len(self.path_X)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_X = Image.open(self.path_X[index % len(self.path_X)]).convert("RGB")
        if self.test == False:
            img_Y = Image.open(self.path_Y[index % len(self.path_Y)]).convert("RGB")

        if self.transform:
            img_X = self.transform(img_X)
            if self.test == False:
                img_Y = self.transform(img_Y)

        if self.test == False:
            return img_X, img_Y
        return img_X
