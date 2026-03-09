import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import v2

class Cifar10Dataset(Dataset):
    def __init__(self, root_directory="", dataset="data/train.csv", transform=None):
        self.root_directory = root_directory
        self.image_directory = pd.read_csv(dataset)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_directory)
    
    def __getitem__(self, idx: int) -> tuple:
        image_path = self.image_directory.iloc[idx, self.image_directory.columns.get_loc("filenames")]
        label = self.image_directory.iloc[idx, self.image_directory.columns.get_loc("labels")]
        data = torch.from_numpy(np.load(self.root_directory + image_path)).reshape((3,32,32))

        if self.transform:
            data = self.transform(data)

        return data, label
    

# dataset sanity check
import matplotlib.pyplot as plt
def sanity_check():
    data = torch.from_numpy(np.load("data/train/woodland_caribou_s_000999.npy")).reshape(3, 32, 32).permute(1,2,0)
    plt.imshow(data.numpy())
    plt.show() 