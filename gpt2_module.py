# gpt2_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from gpt2_model import GPT2Model
from text_dataset import TextDataset
from transformers import GPT2Tokenizer

# Your GPT-2 encoder implementation here

class GPT2Module(pl.LightningModule):
    def __init__(self, vocab_size, num_classes, model_name_or_path, learning_rate=2e-5, batch_size=8):
        super(GPT2Module, self).__init__()
        self.model = GPT2Model(vocab_size)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        src, tgt, labels = batch
        logits = self(src, tgt)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Handle the case where you have a single text sequence
        src = batch

        # Preprocess the text sequence and convert it into the appropriate input format
        input_ids = self.tokenizer.encode(src, return_tensors='pt')

        # Forward pass to get the logits from the GPT-2 model
        logits = self.model(src=input_ids)[0]

        # Compute the targets by shifting the input_ids by one position
        targets = input_ids[:, 1:]

        # Compute the language modeling loss (cross-entropy loss)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        # Return the validation loss or any metrics you want to track
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def prepare_data(self):
        # Implement dataset preparation if needed
        self.train_dataset = TextDataset("babylm_data/babylm_10M", ".train")
        self.val_dataset = TextDataset("babylm_data/babylm_dev", ".dev")
        self.test_dataset = TextDataset("babylm_data/babylm_test", ".test")

    def train_dataloader(self):
        # Implement training DataLoader
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        # Implement validation DataLoader
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        return val_loader

    def test_dataloader(self):
        # Implement testing DataLoader
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        return test_loader