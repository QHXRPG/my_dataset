import torch
import torch.nn as nn
from PIL import Image

class My_data(nn.Module):
    def __init__(self, images_path:list, images_class:list, transform=None):
        super(My_data, self).__init__()
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode!='RGB':
            raise ValueError(f"the images'mode '{img.mode}' is not RGB.")
        labels = self.images_class
        if self.transform is not None:
            img = self.transform(img)
        return img,labels