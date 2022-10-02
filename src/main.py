
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from ModelInterface import ModelInterface
from datamodel.CIFAR10DataModel import CIFAR10DataModule
from pytorch_lightning.callbacks import Callback
import time

if __name__ == '__main__':

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

    logger = CSVLogger("logs", name="resnet32_cifar")
    trainer = pl.Trainer(max_epochs=5, gpus=0, callbacks=[ckpt_callback, EpochCallback()],
                         logger=logger,
                         max_time="00:01:00:00", check_val_every_n_epoch=1,
                         log_every_n_steps=1, strategy="ddp")

    model = ModelInterface("ResNet32")
    dataModule = CIFAR10DataModule()
    trainer.fit(model, dataModule)
    # cli = LightningCLI(ModelInterface.py, CIFAR10DataModule,
    #                     trainer_defaults={"logger": logger}, save_config_overwrite=True)
