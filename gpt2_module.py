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
    def __init__(self, vocab_size, model_name_or_path, learning_rate=2e-5, batch_size=8):
        super(GPT2Module, self).__init__()
        self.model = GPT2Model()
        self.model.to('cuda:0')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def forward(self, src):
        return self.model(src)

    def compute_loss(self, logits, input_ids):
        # Slice off the final prediction from the logits so it matches the targets' shape
        logits = logits[:, :-1, :]

        # Compute the targets by shifting the input_ids by one position
        targets = input_ids[:, 1:]

        # Compute the language modeling loss (cross-entropy loss)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        
        return loss
    
    def training_step(self, batch, batch_idx):
        # Now, each batch item is already a tokenized chunk.
        # Convert the batch (a list of token chunks) into a tensor.
        input_ids = batch.to(self.device)  # Shape: [batch_size, max_length]

        # Forward pass for this chunk
        logits = self(input_ids)

        # Compute the loss for this batch
        loss = self.compute_loss(logits, input_ids)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Each batch item is already a tokenized chunk.
        input_ids = batch.to(self.device)  # Shape: [batch_size, max_length]

        # Forward pass to get the logits from the GPT-2 model
        logits = self(input_ids)

        # Compute the loss for this batch
        loss = self.compute_loss(logits, input_ids)

        # Return the validation loss or any metrics you want to track
        return {'val_loss': loss}



    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def prepare_data(self):
        # Implement dataset preparation if needed
        self.train_dataset = TextDataset("babylm_data/babylm_10M", ".train")
        self.val_dataset = TextDataset("babylm_data/babylm_dev", ".dev")
        self.test_dataset = TextDataset("babylm_data/babylm_test", ".test")

    def train_dataloader(self):
        # Implement training DataLoader
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12)
        return train_loader

    def val_dataloader(self):
        # Implement validation DataLoader
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=12)
        return val_loader

    def test_dataloader(self):
        # Implement testing DataLoader
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=12)
        return test_loader