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
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        
        return loss
    
    def training_step(self, batch, batch_idx):
        max_length = 1024  # GPT-2's max token length
        total_loss = 0.0
        total_chunks = 0

        for text in batch:
            # Convert text to input IDs
            input_ids = self.tokenizer.encode(text, return_tensors='pt').squeeze(0)
            input_ids = input_ids.to(self.device)
            # Split the input_ids into chunks of max_length
            num_chunks = len(input_ids) // max_length + int(len(input_ids) % max_length != 0)

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * max_length
                end_idx = start_idx + max_length

                chunk = input_ids[start_idx:end_idx].unsqueeze(0)  # unsqueeze to add the batch dimension back

                # Forward pass for this chunk
                logits = self(chunk)

                # Compute the loss for this chunk
                # Note: You need to implement the compute_loss method
                loss = self.compute_loss(logits, chunk)
                
                total_loss += loss.item()
                total_chunks += 1

        # Average the losses
        average_loss = total_loss / total_chunks
        self.log('train_loss', average_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return torch.tensor(average_loss, device=self.device)

    def validation_step(self, batch, batch_idx):
        # Handle the case where you have a single text sequence
        src = batch

        # Preprocess the text sequence and convert it into the appropriate input format
        input_ids = self.tokenizer.encode(src, return_tensors='pt')
        input_ids = input_ids.to(self.device)

        # Forward pass to get the logits from the GPT-2 model
        logits = self.model(src=input_ids)

        # Slice off the final prediction from the logits so it matches the targets' shape
        logits = logits[:, :-1, :]

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