# train_gpt2.py
import torch
import pytorch_lightning as pl
from gpt2_module import GPT2Module

def main():
    # Define hyperparameters
    model_name_or_path = 'gpt2'
    vocab_size = 50257  # Replace with the actual vocabulary size of your dataset
    learning_rate = 2e-5
    batch_size = 8
    max_epochs = 5

    # Initialize the GPT-2 model
    model = GPT2Module(vocab_size, model_name_or_path, learning_rate, batch_size)

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator='cpu')
    trainer.fit(model)

if __name__ == "__main__":
    main()
