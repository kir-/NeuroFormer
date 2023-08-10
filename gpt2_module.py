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

# Your GPT-2 encoder implementation here
def custom_weight_init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
                
class GPT2Module(pl.LightningModule):
    
    def __init__(self, vocab_size=50257, model_name_or_path='gpt2', learning_rate=2e-5, batch_size=8):
        super(GPT2Module, self).__init__()
        self.model = GPT2Model()
        self.model.apply(custom_weight_init)
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
        input_ids = batch.to(self.device)
        logits = self(input_ids)
        loss = self.compute_loss(logits, input_ids)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # Define the scheduler
        # max_lr can be set to a value higher than your base learning rate, 
        # pct_start is the fraction of the total training steps that increasing phase occupies
        scheduler = {
            'scheduler': OneCycleLR(optimizer, max_lr=3e-4, total_steps=None, epochs=self.trainer.max_epochs, steps_per_epoch=len(self.train_dataloader()), pct_start=0.3, anneal_strategy='cos', div_factor=25.0, final_div_factor=10000.0),
            'interval': 'step',
        }

        return [optimizer], [scheduler]

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