import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

class TextDataset(Dataset):
    def __init__(self, data_folder, file_extension, max_length=1024):
        self.data_folder = data_folder
        self.file_extension = file_extension
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-small')
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
                    
                    # Split into chunks and add to samples
                    for i in range(0, len(input_ids), self.max_length):
                        end = min(i + self.max_length, len(input_ids))
                        chunk = input_ids[i:end]
                        
                        # Pad the chunk if it's not of max_length
                        if len(chunk) < self.max_length:
                            chunk = torch.cat((chunk, torch.tensor([self.tokenizer.pad_token_id] * (self.max_length - len(chunk)))))
                        
                        samples.append(chunk)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
