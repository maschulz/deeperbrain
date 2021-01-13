import copy
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn import metrics

from data import get_dataloaders
from utils import sanitize_dict


@dataclass
class Config:
    learning_rate: float = 0.01
    channels: Tuple[int, ...] = (32, 64, 128, 256, 256, 64)
    dropout: bool = True
    sample_size: int = 8000
    seed: int = 0
    gpu: str = 0
    batch_size: int = 8
    target: str = 'ageC'
    weight_decay: float = 0.001
    arch: str = 'peng'
    lr_decay: float = 0.3


def train(cfg: Config):
    if cfg.arch == 'peng':
        from peng import PengNet as Model
        epochs = 150
    else:
        raise ValueError

    model = Model(cfg)
    checkpoint_callback = ModelCheckpoint(dirpath=SAVE_PATH, monitor='loss/val',
                                          save_last=True, mode='min')
    trainer = Trainer(
        gpus=str(cfg.gpu),
        logger=WandbLogger(project='peng'),
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)


class BaseNet(pl.LightningModule):
    def __init__(self, hparams: Config):
        super().__init__()
        torch.set_num_threads(15)
        self.hparams = sanitize_dict(copy.copy(vars(hparams)))  # TODO: clean up

        self.dataloaders, self.regression, self.output_dim = get_dataloaders(hparams.sample_size, hparams.seed,
                                                                             hparams.batch_size,
                                                                             hparams.target)

        self.learning_rate = hparams.learning_rate
        self.weight_decay = hparams.weight_decay
        self.channels = hparams.channels
        self.dropout = hparams.dropout
        self.lr_decay = hparams.lr_decay

        # print(self.output_dim)
        # print(self.dropout)

        self.build_model()

    def build_model(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return self.dataloaders['val']

    def test_dataloader(self):
        return self.dataloaders['test']

    def loss(self, x, y, postfix='train'):
        logits = self.forward(x)

        if self.regression:
            logits = logits.view(logits.size()[:1])
            loss = F.mse_loss(logits, y)
            y_pred = logits.detach().cpu()
            y = y.cpu()
            # note that these metrics are only used for diagnostic tensorboard/wandb plots
            r2 = max(metrics.r2_score(y, y_pred), -1)
            mae = min(metrics.mean_absolute_error(y, y_pred), 20)
            mse = min(metrics.mean_squared_error(y, y_pred), 500)
            log = {
                'loss': loss,
                'acc': torch.tensor(r2, dtype=torch.float),
                'mae': torch.tensor(mae, dtype=torch.float),
                'mse': torch.tensor(mse, dtype=torch.float),
            }
        else:
            logits = logits.view(logits.size()[:2])
            loss = F.cross_entropy(logits, y)
            y_pred = logits.argmax(1)
            acc = y_pred.eq(y).float().mean()
            log = {
                'loss': loss,
                'acc': acc
            }

        log = {f'{k}/{postfix}': log[k] for k in log}
        log['lr'] = torch.tensor(self.optimizer.param_groups[0]['lr'])

        return log

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x.float()
        y = y.float()

        loss_dict = self.loss(x, y, postfix='train')
        for k in loss_dict:
            self.log(k, loss_dict[k])
        return loss_dict['loss/train']

    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss_dict = self.loss(x, y, postfix='val')
        for k in loss_dict:
            self.log(k, loss_dict[k])
        return loss_dict['loss/val']

    def configure_optimizers(self):
        raise NotImplementedError
