import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

class TextDataset(Dataset):
    def __init__(self, data_folder, file_extension, max_length=1024):
        self.data_folder = data_folder
        self.file_extension = file_extension
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.max_length = max_length
        self.samples = self.build_samples()

    def build_samples(self):
        samples = []
        for file_name in os.listdir(self.data_folder):
            if file_name.endswith(self.file_extension):
                file_path = os.path.join(self.data_folder, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                    # Tokenize the text
                    input_ids = self.tokenizer.encode(text, return_tensors="pt").squeeze()

                    # Sliding window approach
                    stride = int(self.max_length * 0.8)  # Using 80% of max_length as stride. You can adjust as needed.
                    for i in range(0, len(input_ids) - self.max_length + 1, stride):
                        chunk = input_ids[i:i + self.max_length]
                        samples.append(chunk)

                    # Handle the remaining part of input_ids if any
                    if len(input_ids) > i + self.max_length:
                        chunk = input_ids[-self.max_length:]
                        samples.append(chunk)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
