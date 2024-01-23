import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import os
import numpy as np
from skimage import io, transform
from torchvision.io import read_image

class maldataset(Dataset):
    def __init__(self, csv_file, root_dir,class_index, transform=None):
    
        self.malware_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.class_index=class_index
        self.transform = transform
       
    def __len__(self):
        return len(self.malware_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        img_name = os.path.join(self.root_dir,
                                self.malware_frame.iloc[idx, 1])
    
        image = read_image(img_name)
        target = self.malware_frame.iloc[idx, 2]
        target = self.class_index[target]
      
        if self.transform:
            image = self.transform(image)
        return image,target