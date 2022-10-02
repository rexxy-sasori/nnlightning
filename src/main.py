
import pytorch_lightning as pl
from torch import optim
from pytorch_lightning.loggers import CSVLogger
from ModelInterface import ModelInterface
from datamodel.CIFAR10DataModel import CIFAR10DataModule
from pytorch_lightning.callbacks import Callback
import time


if __name__ == '__main__':

    logger_dir = "logs"
    logger_name = "resnet32_cifar"
    max_epochs = 5
    gpus = 0
    max_time = "00:01:00:00"
    check_val_every_n_epoch = 1
    log_every_n_steps = 1
    strategy = "ddp"
    optimizer = optim.SGD
    lr = 1e-2
    model_name = "ResNet32"

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min'
    )

    class EpochCallback(Callback):

        def on_epoch_start(self, trainer, pl_module):
            pl_module.epochTimeKeeper = time.time()

        def on_epoch_end(self, trainer, pl_module):
            epoch_time = time.time() - pl_module.epochTimeKeeper
            pl_module.log("epoch_time", epoch_time)


    logger = CSVLogger(logger_dir, name=logger_name)
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus, callbacks=[ckpt_callback, EpochCallback(), lr_monitor],
                         logger=logger,
                         max_time=max_time, check_val_every_n_epoch=check_val_every_n_epoch,
                         log_every_n_steps=log_every_n_steps, strategy=strategy)

    model = ModelInterface(model_name=model_name, optimizer_func=optimizer, lr=lr)
    dataModule = CIFAR10DataModule()
    trainer.fit(model, dataModule)
    # cli = LightningCLI(ModelInterface.py, CIFAR10DataModule,
    #                     trainer_defaults={"logger": logger}, save_config_overwrite=True)
