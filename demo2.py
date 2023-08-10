# gpt2_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from gpt2_model import GPT2Model
from text_dataset import TextDataset
from transformers import GPT2Tokenizer
from torch.optim.lr_scheduler import OneCycleLR
dummy_input = torch.randint(0, 50257, (2, 50))  # Assuming batch size of 2 and sequence length of 50
model = GPT2Model()
output = model(dummy_input)
print(output.shape)