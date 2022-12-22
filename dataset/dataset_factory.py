from .preprocessing import get_mapping, preprocess_img

from torch.utils.data import Dataset
import torch
from skimage import io
import numpy as np

import os


class IAMData(Dataset):
    def __init__(self, df):
        self.df = df
        self.char_dict = get_mapping(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name, word = self.df.iloc[idx, 0], self.df.iloc[idx, 1]
        image, word = preprocess_img(img_name, word)

        return image, word


class FormDataset(Dataset):
    def __init__(self, image_folder_path, labels, transform=None, target_transform=None):
        self.image_folder_path = image_folder_path
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self) : 
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder_path, self.labels.iloc[idx]['name'] + '.png')
        image = io.imread(img_path, as_gray=True)
        if self.transform:
            img = self.transform(image)
        
        xmins = self.labels.iloc[idx]['xmin']
        ymins = self.labels.iloc[idx]['ymin']
        xmaxs = self.labels.iloc[idx]['xmax']
        ymaxs = self.labels.iloc[idx]['ymax']
        
        
        boxes = []
        labels = []
        for k in range(len(xmins)) : 
            boxes.append([xmins[k], ymins[k],xmaxs[k],ymaxs[k]])
        
        boxes = torch.as_tensor(np.array(boxes))
            
        target = {}
        target['boxes'] = boxes
            
        
        return img, target