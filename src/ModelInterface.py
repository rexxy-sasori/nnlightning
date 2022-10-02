


import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
import torchmetrics
from typing import Optional
from pytorch_lightning.loggers import CSVLogger
from importlib import import_module


class ModelInterface(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.save_hyperparameters()
        self.load_model(model_name)
        self.lr = 1e-2

    def load_model(self, name):
        try:
            Model = getattr(import_module(
                'model.' + name), name)
        except:
            raise ValueError(
                f'Invalid Module File Name {name}!')
        self.model = Model()

    def configure_optimizers(self):
        optimiser = optim.SGD(self.parameters(), lr=self.lr)
        return optimiser

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = nn.CrossEntropyLoss()(prediction, y)
        acc = torchmetrics.functional.accuracy(prediction, y)
        self.log("training_loss", loss)
        self.log("training_acc", acc)
        self.log("learning-rate", self.lr)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = nn.CrossEntropyLoss()(prediction, y)
        acc = torchmetrics.functional.accuracy(prediction, y)
        self.log("val_loss", loss)
        self.log("val_acc", acc)

