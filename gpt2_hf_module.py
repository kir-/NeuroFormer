import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from text_dataset import TextDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

class GPT2Module(pl.LightningModule):
    def __init__(self, vocab_size=50257, model_name_or_path='gpt2', learning_rate=2e-5, batch_size=8):
        super(GPT2Module, self).__init__()

        config = GPT2Config(vocab_size=vocab_size)
        self.model = GPT2LMHeadModel(config)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

    def forward(self, src):
        return self.model(src).logits

    def compute_loss(self, logits, input_ids):
        # Slice off the final prediction from the logits so it matches the targets' shape
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]  # Compute the targets by shifting the input_ids by one position

        # Compute the language modeling loss (cross-entropy loss)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return loss
    
    def training_step(self, batch, batch_idx):
        input_ids = batch.to(self.device)
        logits = self(input_ids)
        loss = self.compute_loss(logits, input_ids)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch.to(self.device)
        logits = self(input_ids)
        loss = self.compute_loss(logits, input_ids)
        return {'val_loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def prepare_data(self):
        self.train_dataset = TextDataset("babylm_data/babylm_10M", ".train")
        self.val_dataset = TextDataset("babylm_data/babylm_dev", ".dev")
        self.test_dataset = TextDataset("babylm_data/babylm_test", ".test")

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=12)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=12)
        return test_loader
