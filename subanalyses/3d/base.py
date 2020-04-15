import copy
from dataclasses import dataclass
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from sklearn import metrics

from data import get_dataloaders
from utils import torch_stack_dicts, sanitize_dict


@dataclass
class Config:
    learning_rate: float = 0.01
    channels: Tuple[int, ...] = (32, 64, 128, 256, 256, 64)
    dropout: bool = True
    sample_size: int = 8000
    seed: int = 0
    batch_size: int = 8
    target: str = 'sex'
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

    trainer = Trainer(
        # val_check_interval=500,
        max_epochs=epochs,
        default_save_path=f'/home/ms883464/deploy/deeperbrain3/subanalyses/3d/results_new/t_{cfg.target}',
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
            y_pred = logits.detach()
            r2 = max(metrics.r2_score(y_pred, y), -1)  # these are just to clean up the plots..
            mae = min(metrics.mean_absolute_error(y_pred, y), 20)
            mse = min(metrics.mean_squared_error(y_pred, y), 500)
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
        loss_dict = self.loss(x, y, postfix='train')
        return {'loss': loss_dict['loss/train'],
                'progress_bar': {'acc': loss_dict['acc/train']},
                'log': loss_dict}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss_dict = self.loss(x, y, postfix='val')
        return loss_dict

    def validation_epoch_end(self, outputs):
        loss_dict = torch_stack_dicts(outputs)
        return {'progress_bar': {'val_acc': loss_dict['acc/val'], 'val_loss': loss_dict['loss/val']}, 'log': loss_dict}

    def test_step(self, batch, batch_nb):
        x, y = batch
        loss_dict = self.loss(x, y, postfix='test')
        return loss_dict

    def test_epoch_end(self, outputs):
        loss_dict = torch_stack_dicts(outputs)
        return {'progress_bar': {'test_acc': loss_dict['acc/test'], 'test_loss': loss_dict['loss/test']},
                'log': loss_dict}

    def configure_optimizers(self):
        raise NotImplementedError
