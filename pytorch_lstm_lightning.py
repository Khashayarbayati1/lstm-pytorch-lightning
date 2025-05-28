import os
import re

import torch
import torch.nn as nn 
import torch.nn.functional as F # Gives us Activation Functions
from torch.optim import Adam

import lightning as L
from torch.utils.data import TensorDataset, DataLoader

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

import matplotlib.pyplot as plt
import seaborn as sns

class LightningLSTM(L.LightningModule):
    
    def __init__(self):
        
        super().__init__()
        
        L.seed_everything(seed=42)
        
        self.lstm = nn.LSTM(input_size=1, hidden_size=1) # input_size: Number of features, or variables we have in the training data, hidden_size= Number of output values we want
        
    def forward(self, input):
        
        input_trans = input.view(len(input), 1)
        
        lstm_out, temp = self.lstm(input_trans)
        
        prediction = lstm_out[-1]
        return prediction
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)
    
    def training_step(self, batch, batch_idx):
        
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i) ** 2
        
        self.log("train_loss", loss)
        
        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)
        
        return loss
    
    import os
    
class ModelTrainer:
    
    def __init__(self, model, train_inputs, train_labels, current_max_epochs):
        self.model = model
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.current_max_epochs = current_max_epochs
    
    def train(self, trainer_dir="LightningLSTM_logs", model_name="LSTM_model"):
        run_path = os.path.join(trainer_dir, model_name)
        self.trainer = None

        # Prepare dataset and logger
        self.dataset = TensorDataset(self.train_inputs, self.train_labels)
        self.dataloader = DataLoader(self.dataset)
        self.logger = TensorBoardLogger(trainer_dir, name=model_name)

        # No logs? Train from scratch
        if not os.path.exists(run_path):
            print("No previous logs found. Training from scratch.")
            self.trainer = L.Trainer(max_epochs=self.current_max_epochs, log_every_n_steps=2, logger=self.logger)
            self.trainer.fit(self.model, train_dataloaders=self.dataloader)
            return

        # Get latest run version
        version_folders = sorted([f for f in os.listdir(run_path) if f.startswith("version_")])
        if not version_folders:
            print("No version folders found. Training from scratch.")
            self.trainer = L.Trainer(max_epochs=self.current_max_epochs, log_every_n_steps=2, logger=self.logger)
            self.trainer.fit(self.model, train_dataloaders=self.dataloader)
            return

        latest_version = version_folders[-1]
        checkpoint_dir = os.path.join(run_path, latest_version, "checkpoints")

        # No checkpoint folder?
        if not os.path.exists(checkpoint_dir):
            print("No checkpoint directory found. Training from scratch.")
            self.trainer = L.Trainer(max_epochs=self.current_max_epochs, log_every_n_steps=2, logger=self.logger)
            self.trainer.fit(self.model, train_dataloaders=self.dataloader)
            return

        # Get checkpoint files
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
        if not checkpoints:
            print("No checkpoints found. Training from scratch.")
            self.trainer = L.Trainer(max_epochs=self.current_max_epochs, log_every_n_steps=2, logger=self.logger)
            self.trainer.fit(self.model, train_dataloaders=self.dataloader)
            return

        # Pick the first .ckpt (can sort if needed)
        best_checkpoint_path = os.path.join(checkpoint_dir, sorted(checkpoints)[0])
        print(f"Resuming from checkpoint: {best_checkpoint_path}")

        # Extract epoch number
        match = re.search(r"epoch=(\d+)", best_checkpoint_path)
        last_epoch = int(match.group(1)) + 1 if match else 0

        if self.current_max_epochs <= last_epoch:
            print(f"Model already trained to epoch {last_epoch}. No further training required.")
            return

        self.trainer = L.Trainer(max_epochs=self.current_max_epochs, log_every_n_steps=2, logger=self.logger)
        self.trainer.fit(self.model, train_dataloaders=self.dataloader, ckpt_path=best_checkpoint_path)
    
        
if __name__ == "__main__":
    
    model = LightningLSTM()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_inputs = torch.tensor([[0.0, 0.5, 0.25, 1.0], [1.0, 0.5, 0.25, 1.0]], device=device)
    train_labels = torch.tensor([0.0, 1.0], device=device)
    
    model = LightningLSTM().to(device)
    
    current_max_epochs = 300
    trainer_class = ModelTrainer(model, train_inputs, train_labels, current_max_epochs)
    trainer_class.train()
    