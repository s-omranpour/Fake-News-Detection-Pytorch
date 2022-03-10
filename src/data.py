import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

class CSIDataset(Dataset):
    def __init__(self, pkl_dir):
        super().__init__()
        self.samples = []
        for file in os.listdir(pkl_dir):
            sample = pickle.load(open(pkl_dir + file, 'rb'))
            if sample['x_score'].shape[0] > 0:
                self.samples += [sample]
        
    def __getitem__(self, idx):
        data = self.samples[idx]
        return data['x_capture'], data['x_score'], data['label']
    
    def __len__(self):
        return len(self.samples)