from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from typing import Optional

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        # download
        datasets.CIFAR10('data', train=True, download=True, transform=self.transform)

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar10_full = datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.train, self.val = random_split(cifar10_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.predict = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=32)
