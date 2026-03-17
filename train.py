import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from dataset import Cifar10Dataset
from densenet import DenseNetwork

import wandb


class DenseNetworkModel:
    def __init__(self, device,log=True):
        self.model = DenseNetwork().to(device)
        self.device = device
        self.log = log

    def train(self, root_directory, hyperparams, train_transforms, validation_transforms):
        if self.log:
            wandb_log = self.init_logging(hyperparams.batch_size, hyperparams.learning_rate, hyperparams.momentum, hyperparams.weight_decay, hyperparams.num_epochs)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=hyperparams.learning_rate, momentum=hyperparams.momentum, weight_decay=hyperparams.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)
        
        # Create Datasets 
        training_dataset = Cifar10Dataset(root_directory, dataset="data/train.csv", transform=train_transforms)
        train_dataloader = DataLoader(training_dataset, batch_size=hyperparams.batch_size, shuffle=True)
        
        val_dataset = Cifar10Dataset(root_directory, dataset="data/test.csv", transform=validation_transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=hyperparams.batch_size, shuffle=False)

        best_loss = float("inf")
        logging_steps = 0
        for epoch in range(hyperparams.num_epochs):

            # begin training loop 
            self.model.train()
            running_loss = 0.0
            for i, (input, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                assert self.model.training, 'make sure the network is in train mode with `.train()`'

                optimizer.zero_grad() # zero out the gradient from previous batches
                
                input = input.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input) # Predictions

                loss = F.cross_entropy(outputs, labels)

                loss.backward() # backprop step
                optimizer.step() # SGD optimizing step

                running_loss += loss.item()
                if self.log and (i % 10 == 0) and i > 0:
                    wandb_log.log({"eval/loss": loss.item()}, step=logging_steps)
                    logging_steps += 1
            
            top_k=3
            validation_loss, validation_accuracy, validation_top_k_accuracy = self.validation(val_dataset, val_dataloader, top_k) 
            # scheduler.step(validation_loss) # for ReduceLROnPlateau: perform lr reduction if validation loss doesn't improve
            scheduler.step() # for CosineAnnealingLR

            print(f"Epoch: {epoch+1} Training Loss: {running_loss/len(train_dataloader)}")
            print(f" Validation Loss: {validation_loss} Validation Accuracy: {validation_accuracy} Validation Top {top_k} Accuracy {validation_top_k_accuracy}")
            if self.log:
                wandb_log.log({
                    "train/loss": running_loss / len(train_dataloader),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    f"validation_top_{top_k}_accuracy": validation_top_k_accuracy,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
            if validation_loss < best_loss:
                os.makedirs("model_weights", exist_ok=True)
                torch.save(self.model.state_dict(), "model_weights/best_val_loss.pt")
                best_loss = validation_loss

    def validation(self, val_dataset, val_dataloader, top_k=5):
        total_loss = 0
        correct = 0
        top_k_correct = 0

        self.model.eval() # set model to eval mode so the weights won't get changed
        with torch.no_grad():
            for input, labels in tqdm(val_dataloader, desc="Validation Run"):
                input = input.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input) 

                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()

                # get accuracy
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                top_k_preds = torch.topk(outputs, top_k, dim=1).indices
                top_k_correct += (top_k_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

        avg_loss = total_loss / len(val_dataloader)  
        accuracy = correct / len(val_dataset)
        top_k_accuracy = top_k_correct / len(val_dataset)

        return avg_loss, accuracy, top_k_accuracy
    
    def init_logging(self, batch_size, learning_rate, momentum, weight_decay, num_epochs):
        wandb_log = wandb.init(
            entity="wtoth21",
            # Set the wandb project where this run will be logged.
            project="DenseNetwork",
            # Track hyperparameters and run metadata.
            config={
                "architecture": "Dense Network",
                "dataset": "Cifar10",
                "num_dense_layers":4,
                "projection_dimensions":12,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "momentum": momentum,
                "l2 regularization": weight_decay,
                "epochs": num_epochs,
            })
        return wandb_log