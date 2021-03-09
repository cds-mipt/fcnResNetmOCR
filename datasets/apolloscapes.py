import os
import re
import math
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ApolloscapesLaneSegmentation(Dataset):
    def __init__(self, root_dir, image_set='train', train_extra=True, transforms=None):

        self.root_dir = root_dir
        self.image_set = image_set
        self.transforms = transforms

        self.images = []
        self.targets = []
        
        if image_set == 'train':
            with open('list/train.txt', 'r') as f:
                for line in f:
                    img, trg = line.split(';')
                    self.images.append(img)
                    self.targets.append(trg[:-1])

        elif image_set == 'val':
            with open('list/val.txt', 'r') as f:
                for line in f:
                    img, trg = line.split(';')
                    #img = line[:-1]
                    self.images.append(img)
                    self.targets.append(trg[:-1])
                self.images = list(sorted(self.images, key=lambda x: x.split('/')[-1]))
                self.targets = list(sorted(self.targets, key=lambda x: x.split('/')[-1]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        #target = Image.open(self.images[index]).convert('P')
        target = Image.open(self.targets[index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)
            #image = self.transforms(image)

        return image, target

