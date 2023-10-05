# network models for RGAN
# Created: 6/10/2022
# Status: ok

import sys
import math

import torch.nn as nn

import torch

# 2 options:
# (1): put all Gs in one, and directly combine them
# (2): put all in networks, and combine outside


class _MLP_Surv(nn.Module):
    def __init__(self, in_size, drop_rate, fil_num, output_shape=1):
        super(_MLP_Surv, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_size)
        self.bn2 = nn.BatchNorm1d(fil_num)
        self.fc1 = nn.Linear(in_size, fil_num)
        self.fc2 = nn.Linear(fil_num, output_shape)
        self.do1 = nn.Dropout(drop_rate)
        self.do2 = nn.Dropout(drop_rate)
        self.ac1 = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self, X):
        X = self.bn1(X)

        out = self.do1(X)
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.ac1(out)

        out = self.do2(out)
        out = self.fc2(out)
        out = self.sig(out)
        return out


class _Gs_Model(nn.Module):
    def __init__(self, config):
        # this version has 2+ generators. with potential option to share layers and combine outputs
        super().__init__()
        self.config = config
        # optional shared layer

        # generator 1
        self.g1_convt1 = nn.ConvTranspose3d(
            config["in_channels"],
            config["fil_num_g"] * 8,
            (3, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )
        self.g1_bn1 = nn.BatchNorm3d(config["fil_num_g"] * 8)
        self.g1_convt2 = nn.ConvTranspose3d(
            config["fil_num_g"] * 8,
            config["out_channels"],
            (3, 1, 1),
            stride=(2, 1, 1),
            padding=(3, 0, 0),
        )
        self.g1_bn2 = nn.BatchNorm3d(config["out_channels"])

        self.g1_convt3 = nn.ConvTranspose3d(
            config["in_channels"],
            config["in_channels"],
            (3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
        )
        self.g1_bn3 = nn.BatchNorm3d(config["in_channels"])
        self.g1_convt4 = nn.ConvTranspose3d(
            config["in_channels"],
            config["in_channels"],
            (11, 1, 1),
            stride=(2, 1, 1),
            padding=0,
        )
        self.g1_bn4 = nn.BatchNorm3d(config["in_channels"])
        self.g1_convt5 = nn.ConvTranspose3d(
            config["in_channels"],
            config["in_channels"],
            (11, 1, 1),
            stride=(2, 1, 1),
            padding=0,
        )
        self.g1_bn5 = nn.BatchNorm3d(config["in_channels"])
        self.g1_convt6 = nn.ConvTranspose3d(
            config["in_channels"],
            config["in_channels"],
            (11, 1, 1),
            stride=(2, 1, 1),
            padding=0,
        )
        self.g1_bn6 = nn.BatchNorm3d(config["in_channels"])

        # generator 2
        self.g2_convt1 = nn.ConvTranspose3d(
            config["in_channels"],
            config["fil_num_g"] * 8,
            (3, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )
        self.g2_bn1 = nn.BatchNorm3d(config["fil_num_g"] * 8)
        self.g2_convt2 = nn.ConvTranspose3d(
            config["fil_num_g"] * 8,
            config["out_channels"],
            (3, 1, 1),
            stride=(2, 1, 1),
            padding=(3, 0, 0),
        )
        self.g2_bn2 = nn.BatchNorm3d(config["out_channels"])

        self.a = nn.ReLU()
        self.ao = nn.Tanh()
        # self.a = nn.Sigmoid()

    def forward(self, x):
        # print('first_g1', x.shape)
        o1 = self.g1_convt1(x)
        o1 = self.g1_bn1(o1)
        o1 = self.a(o1)
        # print(x.shape)
        o1 = self.g1_convt2(o1)
        o1 = self.g1_bn2(o1)
        o1 = self.ao(o1)
        # print(o1.shape)
        # sys.exit()

        # print('first_g2', x.shape)
        o2 = self.g2_convt1(x)
        o2 = self.g2_bn1(o2)
        o2 = self.a(o2)
        # print(x.shape)
        o2 = self.g2_convt2(o2)
        o2 = self.g2_bn2(o2)
        o2 = self.ao(o2)
        # print(o2.shape)

        # optinal weighted combination
        o = o1 + o2
        # sys.exit()

        # x = self.convt3(x)
        # x = self.bn3(x)
        # x = self.ao(x)
        # print(x.shape)
        # x = self.convt4(x)
        # x = self.bn4(x)
        # x = self.a(x)
        # print(x.shape)
        # x = self.convt5(x)
        # x = self.bn5(x)
        # x = self.a(x)
        # print(x.shape)
        # x = self.convt6(x)
        # x = self.bn6(x)
        # x = self.ao(x)
        # print('last first', x.shape)
        # sys.exit()

        return o


class _G_Model(nn.Module):
    # this version has only 1 generator, so take a few slices and produce entire brain.
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.convt1 = nn.Conv3d(
            config["in_channels"],
            config["fil_num_g"] * 4,
            (3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
        )
        self.bn1 = nn.BatchNorm3d(config["fil_num_g"] * 4)
        self.convt2 = nn.Conv3d(
            config["fil_num_g"] * 4,
            config["fil_num_g"] * 8,
            (3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
        )
        self.bn2 = nn.BatchNorm3d(config["fil_num_g"] * 8)

        self.convt3 = nn.ConvTranspose3d(
            config["fil_num_g"] * 8,
            config["fil_num_g"] * 4,
            (3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
        )
        self.bn3 = nn.BatchNorm3d(config["fil_num_g"] * 4)
        self.convt4 = nn.ConvTranspose3d(
            config["fil_num_g"] * 4,
            config["out_channels"],
            (3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
        )
        self.bn4 = nn.BatchNorm3d(config["out_channels"])

        # self.convt1 = nn.ConvTranspose3d(config['in_channels'], config['fil_num_g']*8, (3, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        # self.bn1 = nn.BatchNorm3d(config['fil_num_g']*8)
        # self.convt2 = nn.ConvTranspose3d(config['fil_num_g']*8, config['out_channels'], (3, 1, 1), stride=(2, 1, 1), padding=(3, 0, 0))
        # self.bn2 = nn.BatchNorm3d(config['out_channels'])

        # self.convt1 = nn.ConvTranspose3d(config['in_channels'], config['fil_num_g']*4, (3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        # self.bn1 = nn.BatchNorm3d(config['fil_num_g']*4)

        # self.convt2 = nn.ConvTranspose3d(config['fil_num_g']*4, config['fil_num_g']*2, (3, 1, 1), stride=(2, 1, 1), padding=(4, 0, 0))
        # self.bn2 = nn.BatchNorm3d(config['fil_num_g']*2)

        # self.convt3 = nn.ConvTranspose3d(config['fil_num_g']*2, config['fil_num_g']*1, (3, 1, 1), stride=(2, 1, 1), padding=(4, 0, 0))
        # self.bn3 = nn.BatchNorm3d(config['fil_num_g']*1)

        # self.convt4 = nn.ConvTranspose3d(config['fil_num_g']*1, config['out_channels'], (3, 1, 1), stride=(2, 1, 1), padding=(3, 0, 0))
        # self.bn4 = nn.BatchNorm3d(config['out_channels'])

        # self.convt4 = nn.ConvTranspose3d(config['in_channels'], config['in_channels'], (11, 1, 1), stride=(2, 1, 1), padding=0)
        # self.bn4 = nn.BatchNorm3d(config['in_channels'])

        self.convt5 = nn.ConvTranspose3d(
            config["in_channels"],
            config["in_channels"],
            (11, 1, 1),
            stride=(2, 1, 1),
            padding=0,
        )
        self.bn5 = nn.BatchNorm3d(config["in_channels"])
        self.convt6 = nn.ConvTranspose3d(
            config["in_channels"],
            config["in_channels"],
            (11, 1, 1),
            stride=(2, 1, 1),
            padding=0,
        )
        self.bn6 = nn.BatchNorm3d(config["in_channels"])

        self.a = nn.ReLU()
        self.ao = nn.Tanh()
        # self.a = nn.Sigmoid()

    def forward(self, x):
        # print('first', x.shape)
        x = self.convt1(x)
        x = self.bn1(x)
        x = self.a(x)
        # print(x.shape)
        x = self.convt2(x)
        x = self.bn2(x)
        x = self.a(x)
        # print(x.shape)
        x = self.convt3(x)
        x = self.bn3(x)
        x = self.a(x)
        # print(x.shape)
        x = self.convt4(x)
        x = self.bn4(x)
        x = self.ao(x)
        # print(x.shape)
        # sys.exit()
        # x = self.convt5(x)
        # x = self.bn5(x)
        # x = self.a(x)
        # print(x.shape)
        # x = self.convt6(x)
        # x = self.bn6(x)
        # x = self.ao(x)
        # print('last first', x.shape)
        # sys.exit()

        return x


class _D_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        self.conv1 = nn.Conv3d(
            config["out_channels"],
            config["out_channels"],
            config["kernel_size"],
            stride=3,
            padding=1,
        )  # the D only takes entire 3D as input
        self.bn1 = nn.BatchNorm3d(config["out_channels"])
        self.conv2 = nn.Conv3d(
            config["out_channels"],
            config["out_channels"],
            config["kernel_size"],
            stride=3,
            padding=1,
        )  # the D only takes entire 3D as input
        self.bn2 = nn.BatchNorm3d(config["out_channels"])
        self.conv3 = nn.Conv3d(
            config["out_channels"],
            config["out_channels"],
            config["kernel_size"],
            stride=3,
            padding=1,
        )  # the D only takes entire 3D as input
        self.bn3 = nn.BatchNorm3d(config["out_channels"])
        self.conv4 = nn.Conv3d(
            config["out_channels"],
            config["out_channels"],
            (5, 6, 5),
            stride=1,
            padding=0,
        )  # the D only takes entire 3D as input
        self.bn4 = nn.BatchNorm3d(config["out_channels"])

        # self.a = nn.ReLU()
        self.a = nn.LeakyReLU()
        self.ao = nn.Sigmoid()

    def forward(self, x):
        # print('first', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.a(x)
        # print('second', x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.a(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.a(x)
        # print('last first', x.shape)
        x = self.conv4(x)
        x = self.ao(x)
        # print('last first', x.shape)
        # sys.exit()

        return x


class _CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        self.conv1 = nn.Conv3d(
            config["in_channels"],
            config["fil_num"],
            config["kernel_size"],
            stride=2,
            padding=1,
        )  # the CNN takes entire 3D as input
        self.bn1 = nn.BatchNorm3d(config["fil_num"])
        self.conv2 = nn.Conv3d(
            config["fil_num"],
            config["fil_num"] * 2,
            config["kernel_size"],
            stride=2,
            padding=1,
        )
        self.bn2 = nn.BatchNorm3d(config["fil_num"] * 2)
        self.conv3 = nn.Conv3d(
            config["fil_num"] * 2,
            config["fil_num"] * 4,
            config["kernel_size"],
            stride=2,
            padding=1,
        )
        self.bn3 = nn.BatchNorm3d(config["fil_num"] * 4)
        self.conv4 = nn.Conv3d(
            config["fil_num"] * 4,
            config["fil_num"] * 8,
            config["kernel_size"],
            stride=2,
            padding=1,
        )
        self.bn4 = nn.BatchNorm3d(config["fil_num"] * 8)
        self.conv5 = nn.Conv3d(
            config["fil_num"] * 8,
            config["fil_num"] * 16,
            config["kernel_size"],
            stride=2,
            padding=1,
        )
        self.bn5 = nn.BatchNorm3d(config["fil_num"] * 16)
        self.conv6 = nn.Conv3d(
            config["fil_num"] * 16, 1, (4, 5, 4), stride=1, padding=0
        )
        self.bn6 = nn.BatchNorm3d(1)

        # self.a = nn.ReLU()
        self.dr1 = nn.Dropout(config["dropout"])
        self.a1 = nn.LeakyReLU()

        self.dr2 = nn.Dropout(config["dropout"])
        self.a2 = nn.LeakyReLU()

        self.dr3 = nn.Dropout(config["dropout"])
        self.a3 = nn.LeakyReLU()

        self.dr4 = nn.Dropout(config["dropout"])
        self.a4 = nn.LeakyReLU()

        self.dr5 = nn.Dropout(config["dropout"])
        self.a5 = nn.LeakyReLU()
        # self.dr = nn.Dropout(config["dropout"]) #not for shap
        # self.a = nn.LeakyReLU()
        self.ao = nn.Sigmoid()

    def forward(self, x):
        # print('first', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dr1(self.a1(x))

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dr2(self.a2(x))

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.dr3(self.a3(x))

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.dr4(self.a4(x))

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.dr5(self.a5(x))

        # print('last first', x.shape)
        x = self.conv6(x)
        # x = self.bn6(x)
        x = self.ao(x)
        # print('last first', x.shape)
        # print('last first', x.squeeze().shape)
        # sys.exit()

        return x
