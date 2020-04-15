"""
from https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain

Accurate brain age prediction with lightweight deep neural networks
Han Peng, Weikang Gong, Christian F. Beckmann, Andrea Vedaldi, Stephen M Smith bioRxiv 2019.12.17.879346
"""

import torch
import torch.nn as nn

from base import BaseNet


class PengNet(BaseNet):
    def build_model(self):
        n_layer = len(self.channels)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = self.channels[i - 1]
            out_channel = self.channels[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.classifier = nn.Sequential()
        # avg_shape = [5, 6, 5]
        # self.classifier.add_module('average_pool', nn.AvgPool3d(out_channel))
        self.classifier.add_module('average_pool', nn.AdaptiveAvgPool3d((1, 1, 1)))
        if self.dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        i = n_layer
        in_channel = self.channels[-1]
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, self.output_dim, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=self.lr_decay),
            'interval': 'epoch'}
        return [self.optimizer, ], [self.scheduler, ]
