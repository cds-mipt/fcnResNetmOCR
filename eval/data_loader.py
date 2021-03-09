import torch
import numpy as np
import os
from PIL import Image
from config import PALETTE

class LabelDataset(torch.utils.data.Dataset):

    def __init__(self, label_root_dir, flag_convert):
        super(LabelDataset, self).__init__()
        
        self.label_root_dir = label_root_dir
        self.flag_convert = flag_convert
        self.palette = PALETTE
        self.mask_paths = os.listdir(label_root_dir)
        self.mask_paths.sort()

    def __len__(self):
        return len(self.mask_paths)
    
    def convert(self, img):
        arr_3d = np.array(img)
        height = arr_3d.shape[0]
        width = arr_3d.shape[1]
        arr_2d = np.zeros((height, width), dtype=np.uint8)

        for c, i in self.palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i

        return arr_2d
    
    def __getitem__(self, index):
        mask_path = os.path.join(self.label_root_dir, self.mask_paths[index])
        if self.flag_convert: 
            label_img_3d = Image.open(mask_path).convert("RGB")
            label_2d = self.convert(label_img_3d) 
        else:
            label_2d = np.array(Image.open(mask_path))
            
        return torch.LongTensor(label_2d.astype(np.int64))