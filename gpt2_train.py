# train_gpt2.py
import torch
import pytorch_lightning as pl
from gpt2_module import GPT2Module
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def main():
    # Define hyperparameters
    model_name_or_path = 'gpt2'
    vocab_size = 50257  # Replace with the actual vocabulary size of your dataset
    learning_rate = 2e-4
    batch_size = 8
    max_epochs = 10

    checkpoint_callback = ModelCheckpoint(
        dirpath='./',
        filename='model-{epoch}',  # Saves a file like model-epoch=0.ckpt, model-epoch=1.ckpt, etc.
        save_top_k=-1,  # This will save a checkpoint at every epoch
        verbose=True,
    )
    # Initialize the GPT-2 model
    model = GPT2Module(vocab_size, model_name_or_path, learning_rate, batch_size)
  
    logger = TensorBoardLogger("logs", name="gpt2_model")
    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(logger=logger,log_every_n_steps=1,max_epochs=max_epochs,callbacks=[checkpoint_callback])
    trainer.fit(model)
    checkpoint_path = "./model.ckpt"
    trainer.save_checkpoint(checkpoint_path)

if __name__ == "__main__":
    main()
