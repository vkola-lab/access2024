# network models for RGAN
# Created: 6/16/2021
# Status: ok

import sys
import math

import torch.nn as nn

import torch

# 2 options:
# (1): put all Gs in one, and directly combine them
# (2): put all in networks, and combine outside

class _Gs_Model(nn.Module):
    def __init__(self, config):
        #this version has 2+ generators. with potential option to share layers and combine outputs
        super().__init__()
        self.config = config
        # optional shared layer

        # generator 1
        self.g1_convt1 = nn.ConvTranspose3d(config['in_channels'], config['in_channels']*8, (3, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.g1_bn1 = nn.BatchNorm3d(config['in_channels']*8)
        self.g1_convt2 = nn.ConvTranspose3d(config['in_channels']*8, config['out_channels'], (3, 1, 1), stride=(2, 1, 1), padding=(3, 0, 0))
        self.g1_bn2 = nn.BatchNorm3d(config['out_channels'])

        self.g1_convt3 = nn.ConvTranspose3d(config['in_channels'], config['in_channels'], (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.g1_bn3 = nn.BatchNorm3d(config['in_channels'])
        self.g1_convt4 = nn.ConvTranspose3d(config['in_channels'], config['in_channels'], (11, 1, 1), stride=(2, 1, 1), padding=0)
        self.g1_bn4 = nn.BatchNorm3d(config['in_channels'])
        self.g1_convt5 = nn.ConvTranspose3d(config['in_channels'], config['in_channels'], (11, 1, 1), stride=(2, 1, 1), padding=0)
        self.g1_bn5 = nn.BatchNorm3d(config['in_channels'])
        self.g1_convt6 = nn.ConvTranspose3d(config['in_channels'], config['in_channels'], (11, 1, 1), stride=(2, 1, 1), padding=0)
        self.g1_bn6 = nn.BatchNorm3d(config['in_channels'])

        # generator 2
        self.g2_convt1 = nn.ConvTranspose3d(config['in_channels'], config['in_channels']*8, (3, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.g2_bn1 = nn.BatchNorm3d(config['in_channels']*8)
        self.g2_convt2 = nn.ConvTranspose3d(config['in_channels']*8, config['out_channels'], (3, 1, 1), stride=(2, 1, 1), padding=(3, 0, 0))
        self.g2_bn2 = nn.BatchNorm3d(config['out_channels'])

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
    #this version has only 1 generator, so take a few slices and produce entire brain.
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.convt1 = nn.ConvTranspose3d(config['in_channels'], config['in_channels']*8, (3, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.bn1 = nn.BatchNorm3d(config['in_channels']*8)
        self.convt2 = nn.ConvTranspose3d(config['in_channels']*8, config['out_channels'], (3, 1, 1), stride=(2, 1, 1), padding=(3, 0, 0))
        self.bn2 = nn.BatchNorm3d(config['out_channels'])
        self.convt3 = nn.ConvTranspose3d(config['in_channels'], config['in_channels'], (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.bn3 = nn.BatchNorm3d(config['in_channels'])
        self.convt4 = nn.ConvTranspose3d(config['in_channels'], config['in_channels'], (11, 1, 1), stride=(2, 1, 1), padding=0)
        self.bn4 = nn.BatchNorm3d(config['in_channels'])
        self.convt5 = nn.ConvTranspose3d(config['in_channels'], config['in_channels'], (11, 1, 1), stride=(2, 1, 1), padding=0)
        self.bn5 = nn.BatchNorm3d(config['in_channels'])
        self.convt6 = nn.ConvTranspose3d(config['in_channels'], config['in_channels'], (11, 1, 1), stride=(2, 1, 1), padding=0)
        self.bn6 = nn.BatchNorm3d(config['in_channels'])

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
        x = self.ao(x)
        # print(x.shape)
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


        return x

class _D_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        self.conv1 = nn.Conv3d(config['out_channels'], config['out_channels'], config['kernel_size'], stride=3, padding=1)# the D only takes entire 3D as input
        self.bn1 = nn.BatchNorm3d(config['out_channels'])
        self.conv2 = nn.Conv3d(config['out_channels'], config['out_channels'], config['kernel_size'], stride=3, padding=1)# the D only takes entire 3D as input
        self.bn2 = nn.BatchNorm3d(config['out_channels'])
        self.conv3 = nn.Conv3d(config['out_channels'], config['out_channels'], config['kernel_size'], stride=3, padding=1)# the D only takes entire 3D as input
        self.bn3 = nn.BatchNorm3d(config['out_channels'])
        self.conv4 = nn.Conv3d(config['out_channels'], config['out_channels'], (5, 6, 5), stride=1, padding=0)# the D only takes entire 3D as input
        self.bn4 = nn.BatchNorm3d(config['out_channels'])

        # self.a = nn.ReLU()
        self.a = nn.LeakyReLU()
        self.ao = nn.Sigmoid()

    def forward(self, x):
        # print('first', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.a(x)
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

class _Encoder(nn.Module):
    "Encoder that encodes Scan to vector"
    def __init__(self, drop_rate, fil_num=64, in_channels=1, out_channels=1):
        super().__init__()
        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        # MaxPool3d(kernel_size, stride=None, padding=0,...)
        self.conv1 = nn.Conv3d(in_channels, fil_num, 4, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm3d(fil_num)
        self.conv2 = nn.Conv3d(fil_num, 2*fil_num, 4, 2, 0, bias=False)
        self.bn2 = nn.BatchNorm3d(2*fil_num)
        self.conv3 = nn.Conv3d(2*fil_num, 4*fil_num, 4, 2, 0, bias=False)
        self.bn3 = nn.BatchNorm3d(4*fil_num)
        self.conv4 = nn.Conv3d(4*fil_num, 8*fil_num, 4, 2, 0, bias=False)
        self.bn4 = nn.BatchNorm3d(8*fil_num)

        # (D−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        # self.conv5 = nn.ConvTranspose3d(fil_num, in_channels, 4, 2, 0, bias=False, output_padding=(1,1,1))
        # self.bn5 = nn.BatchNorm3d(in_channels)

        self.conva = nn.LeakyReLU()
        self.dr = nn.Dropout(drop_rate)

        self.feature_length = 8*fil_num*5*7*5
        self.l1 = nn.Linear(self.feature_length, 100)
        self.l2 = nn.Linear(100, out_channels)
        self.l1a = nn.LeakyReLU()
        self.l2a = nn.Sigmoid()
        '''
        RELU & Softmax: not learning
        Sigmoid & LeakyReLU & GELU: weak learning
        '''
        # self.ac = nn.GELU()

        self.num = fil_num

    def forward(self, x, stage='train'):
        # print('target', x.shape)
        x = self.dr(self.conva(self.bn1(self.conv1(x))))
        # print('input', x.shape)
        # x = self.dr(self.conva(self.bn5(self.conv5(x))))
        # print('output', x.shape)
        # sys.exit()
        x = self.dr(self.conva(self.bn2(self.conv2(x))))
        x = self.dr(self.conva(self.bn3(self.conv3(x))))
        x = self.dr(self.conva(self.bn4(self.conv4(x))))

        x = x.view(-1, self.feature_length)
        x = self.l1a(self.l1(x))
        x = self.l2a(self.l2(x))
        # x = self.ac(x)
        return x


    # def __init__(self, in_size, drop_rate, fil_num, out):
    #     super().__init__()
    #     self.en_fc1 = nn.Linear(in_size, fil_num)
    #     self.en_fc2 = nn.Linear(fil_num, out)
    #     self.en_bn1 = nn.BatchNorm1d(fil_num)
    #     self.en_bn2 = nn.BatchNorm1d(out)
    #     self.en_do1 = nn.Dropout(drop_rate)
    #     self.en_do2 = nn.Dropout(drop_rate)
    #     self.en_ac1 = nn.ReLU()
    #     # self.en_ac1 = nn.LeakyReLU()
    #     self.en_ac2 = nn.Sigmoid()
    #
    # def forward(self, X):
    #     # out = self.en_do1(out)
    #     out = self.en_fc1(X)
    #     out = self.en_bn1(out)
    #     out = self.en_ac1(out)
    #
    #     # out = self.en_do2(out)
    #     out = self.en_fc2(out)
    #     out = self.en_bn2(out)
    #     out = self.en_ac2(out)
    #
    #     return out

class _Decoder(nn.Module):
    "Decoder that decodes vector to Scan"
    def __init__(self, drop_rate, fil_num, in_channels=1, out_channels=1):
        super().__init__()
        # ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        self.conv1 = nn.ConvTranspose3d(8*fil_num, 4*fil_num, 4, 2, 0, bias=False, output_padding=(1,0,1))
        self.bn1 = nn.BatchNorm3d(4*fil_num)
        self.conv2 = nn.ConvTranspose3d(4*fil_num, 2*fil_num, 4, 2, 0, bias=False, output_padding=(0,0,0))
        self.bn2 = nn.BatchNorm3d(2*fil_num)
        self.conv3 = nn.ConvTranspose3d(2*fil_num, fil_num, 4, 2, 0, bias=False, output_padding=(1,1,1))
        self.bn3 = nn.BatchNorm3d(fil_num)
        self.conv4 = nn.ConvTranspose3d(fil_num, out_channels, 4, 2, 0, bias=False, output_padding=(1,1,1))
        self.bn4 = nn.BatchNorm3d(out_channels)

        self.conva = nn.LeakyReLU()
        self.dr = nn.Dropout(drop_rate)

        self.feature_length = 8*fil_num*5*7*5
        self.l1 = nn.Linear(in_channels, 100)
        self.l2 = nn.Linear(100, self.feature_length)
        self.l1a = nn.Sigmoid()
        self.l2a = nn.LeakyReLU()
        ''' TO TRY
        RELU & Softmax: not learning
        Sigmoid & LeakyReLU & GELU: weak learning
        '''
        # self.ac = nn.GELU()

        self.num = fil_num

    def forward(self, x, stage='train'):
        x = self.l1a(self.l1(x))
        x = self.l2a(self.l2(x))
        x = x.view(-1, 8*self.num,5,7,5)

        x = self.dr(self.conva(self.bn1(self.conv1(x))))
        x = self.dr(self.conva(self.bn2(self.conv2(x))))
        x = self.dr(self.conva(self.bn3(self.conv3(x))))
        x = self.dr(self.conva(self.bn4(self.conv4(x))))

        return x

    # def __init__(self, in_size, drop_rate, fil_num, out):
    #     super().__init__()
    #     self.de_fc1 = nn.Linear(in_size, fil_num)
    #     self.de_fc2 = nn.Linear(fil_num, out)
    #     self.de_bn1 = nn.BatchNorm1d(fil_num)
    #     self.de_bn2 = nn.BatchNorm1d(out)
    #     self.de_do1 = nn.Dropout(drop_rate)
    #     self.de_do2 = nn.Dropout(drop_rate)
    #     self.de_ac1 = nn.ReLU()
    #     # self.de_ac1 = nn.LeakyReLU()
    #     self.de_ac2 = nn.Sigmoid()
    #     '''
    #     RELU & Softmax & Sigmoid: not learning
    #     LeakyReLU: learning, not similar to without it
    #     GELU: best currently
    #     '''
    #
    # def forward(self, X):
    #     # out = self.de_do1(out)
    #     out = self.de_fc1(X)
    #     out = self.de_bn1(out)
    #     out = self.de_ac1(out)
    #
    #     # out = self.de_do2(out)
    #     out = self.de_fc2(out)
    #     out = self.de_bn2(out)
    #     out = self.de_ac2(out)
    #
    #     return out

class _ViT_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        t_layer = nn.TransformerEncoderLayer(d_model=config['emsize'], nhead=config['nhead'], dim_feedforward=config['dim_feedforward'], dropout=config['dropout'])
        self.t_encoder = nn.TransformerEncoder(encoder_layer=t_layer, num_layers=config['nlayers'])
        self.map1 = nn.Linear(60*145*121, config['emsize']) #for now sequence is only 1, will change to 27 once this runs||||||||||||||
        self.map2 = nn.Linear(61*145*121, config['emsize']) #for now sequence is only 1, will change to 27 once this runs||||||||||||||
        # self.embed3 = nn.Embedding(100, config['emsize']) #for now sequence is only 1, will change to 27 once this runs||||||||||||||
        self.pos_encoder1 = PositionalEncoding1(config['seq_len'], config['emsize'])
        # self.pos_encoder2 = PositionalEncoding2(config['emsize'])
        self.t_decoder = nn.Linear(config['emsize'], config['out_dim'])
        self.da = nn.Sigmoid()

    def forward(self, x):
        b_size = x.shape[0]
        x1 = x[:,:,:60].flatten().view(b_size, -1) #in here shape[0] is the batch size
        x2 = x[:,:,60:].flatten().view(b_size, -1) #in here shape[0] is the batch size
        # x3 = x[:,:,60,60,60:65].flatten().view(b_size, -1) #in here shape[0] is the batch size
        # print('input', x3.shape)
        # x3 = self.embed3(x3)
        # print('Embedding, before T-encoder', x3.shape)
        # print('v1', self.t_decoder(x3).shape)
        # print('v2', self.t_decoder(x3.view(b_size, -1)).shape)

        x1 = self.map1(x1)
        x2 = self.map2(x2)
        x = torch.stack((x1, x2))
        # seq_len = 2
        # x = x.view(seq_len, b_size, -1)
        x = self.pos_encoder1(x)
        # x = self.pos_encoder2(x)
        x = self.t_encoder(x)
        # print(x.shape)
        # sys.exit()
        # x = x.view(b_size, -1)
        # print('before linear', x.shape)
        # print(self.t_decoder)
        x = self.t_decoder(x)
        x = self.da(x)
        # print('last', x.shape)
        x = x[0, :]
        # print('last first', x.shape)
        # sys.exit()


        return x

class PositionalEncoding1(nn.Module):
    def __init__(self, seq_len, emb_dim, dropout_rate=0.1):
        super().__init__()
        # super(PositionalEncoding, self).__init__()
        # print(1, num_patches + 1, emb_dim)
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, 1, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding
        # print('self.pos_embedding')
        # print(self.pos_embedding.shape)
        # print(x.shape)
        # sys.exit()

        if self.dropout:
            out = self.dropout(out)
        return out

class PositionalEncoding2(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        # super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('pos')
        # print(self.pe.shape)
        # print(self.pe[:x.size(0), :].shape)
        # print(self.pe[:x.size(0), :])
        x = x + self.pe[:x.size(0), :]
        # print(x.shape)
        # sys.exit()
        return self.dropout(x)

class _CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        self.conv1 = nn.Conv3d(config['in_channels'], config['fil_num'], config['kernel_size'], stride=2, padding=1)# the CNN takes entire 3D as input
        self.bn1 = nn.BatchNorm3d(config['fil_num'])
        self.conv2 = nn.Conv3d(config['fil_num'], config['fil_num']*2, config['kernel_size'], stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(config['fil_num']*2)
        self.conv3 = nn.Conv3d(config['fil_num']*2, config['fil_num']*4, config['kernel_size'], stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(config['fil_num']*4)
        self.conv4 = nn.Conv3d(config['fil_num']*4, config['fil_num']*8, config['kernel_size'], stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(config['fil_num']*8)
        self.conv5 = nn.Conv3d(config['fil_num']*8, config['fil_num']*16, config['kernel_size'], stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(config['fil_num']*16)
        self.conv6 = nn.Conv3d(config['fil_num']*16, 1, (4, 5, 4), stride=1, padding=0)
        self.bn6 = nn.BatchNorm3d(1)

        # self.a = nn.ReLU()
        self.dr = nn.Dropout(config['dropout'])
        self.a = nn.LeakyReLU()
        self.ao = nn.Sigmoid()

    def forward(self, x):
        # print('first', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dr(self.a(x))
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dr(self.a(x))
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.dr(self.a(x))
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.dr(self.a(x))
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.dr(self.a(x))
        # print('last first', x.shape)
        x = self.conv6(x)
        # x = self.bn6(x)
        x = self.ao(x)
        # print('last first', x.shape)
        # print('last first', x.squeeze().shape)
        # sys.exit()

        return x

class _CNN_Surv_Res(nn.Module):
    # The Model that will be used for base model
    def __init__(self, drop_rate, fil_num=64, in_channels=1, out_channels=1):
        super().__init__()
        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        # MaxPool3d(kernel_size, stride=None, padding=0,...)
        # ResNet introduction!!!!!!!!!!
        self.conv1 = nn.Conv3d(in_channels, fil_num, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(fil_num)
        self.conv2 = nn.Conv3d(fil_num, 2*fil_num, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(2*fil_num)
        self.conv3 = nn.Conv3d(2*fil_num, 4*fil_num, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(4*fil_num)
        self.conv4 = nn.Conv3d(4*fil_num, 8*fil_num, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm3d(8*fil_num)
        self.conv5 = nn.Conv3d(8*fil_num, 16*fil_num, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm3d(16*fil_num)
        # torch.nn.init.uniform_(self.conv4.weight)

        self.res_block1 = nn.Conv3d(in_channels, 4*fil_num, 1, 4, 1, bias=False)
        self.res_block2 = nn.Conv3d(4*fil_num, 16*fil_num, 1, 4, 1, bias=False)
        # self.res_block3 = nn.Conv3d(8*fil_num, 16*fil_num, 1, 2, 0, bias=False)

        # (D−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        # self.conv5 = nn.ConvTranspose3d(fil_num, in_channels, 4, 2, 0, bias=False, output_padding=(1,1,1))
        # self.bn5 = nn.BatchNorm3d(in_channels)

        self.conva = nn.LeakyReLU()
        self.dr = nn.Dropout(drop_rate)
        self.mp = nn.MaxPool3d(2, padding=1)

        # self.feature_length = 16*fil_num*1*2*1
        self.feature_length = 16*fil_num*5*6*5
        self.l1 = nn.Linear(self.feature_length, 50)
        self.l2 = nn.Linear(50, out_channels)
        self.l1a = nn.LeakyReLU()
        self.l2a = nn.Sigmoid()
        # self.l2a = nn.Tanh()
        '''
        RELU & Softmax: not learning
        Sigmoid & LeakyReLU & GELU: weak learning
        '''
        # self.ac = nn.GELU()

        self.num = fil_num

    def forward(self, x, stage='train'):
        res = x
        # print('1')
        # print(x.shape)
        x = self.conva(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = self.dr(self.mp(x))
        # print(x.shape)
        # print('2')
        x = self.conva(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = self.dr(self.mp(x))
        # print(x.shape)
        res = self.res_block1(res)
        # print('res', res.shape)
        # print('3')
        x = self.conva(self.bn3(self.conv3(x)+res))
        # x = self.conva(self.bn3(self.conv3(x)))
        # print(x.shape)
        x = self.dr(self.mp(x))
        # print(x.shape)
        # print('4')
        x = self.conva(self.bn4(self.conv4(x)))
        # print(x.shape)
        x = self.dr(self.mp(x))
        # print(x.shape)
        res = self.res_block2(res)
        # print('res', res.shape)
        # print('5')
        x = self.conva(self.bn5(self.conv5(x)+res))
        # x = self.conva(self.bn5(self.conv5(x)))
        x = self.dr(self.mp(x))
        # print(x.shape)
        # sys.exit()

        x = x.view(-1, self.feature_length)
        x = self.l1a(self.l1(x))
        # x = self.l2(x)
        x = self.l2a(self.l2(x))
        # x = self.ac(x)
        return x
