import os
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data_folder, file_extension):
        self.data_folder = data_folder
        self.file_extension = file_extension
        self.samples = self.build_samples()

    def build_samples(self):
        samples = []
        for file_name in os.listdir(self.data_folder):
            if file_name.endswith(self.file_extension):
                file_path = os.path.join(self.data_folder, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    samples.append(text)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


