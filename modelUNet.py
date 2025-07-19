import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MLPLayer(nn.Module):
    def __init__(self, in_size,
                 hidden_arch=[128, 512, 1024],
                 output_size=None,
                 activation=nn.PReLU(),
                 batch_norm=True):

        super(MLPLayer, self).__init__()
        self.in_size = in_size
        self.output_size = output_size
        layer_sizes = [in_size] + [x for x in hidden_arch]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)

            if batch_norm and i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i + 1])
                self.layers.append(bn)

            self.layers.append(activation)

        if output_size is not None:
            layer = nn.Linear(layer_sizes[-1], output_size)
            self.layers.append(layer)
            self.layers.append(activation)

        self.init_weights()
        self.mlp_network = nn.Sequential(*self.layers)

    def forward(self, z):
        return self.mlp_network(z)

    def init_weights(self):
        for layer in self.layers:
            try:
                if isinstance(layer, nn.Linear):
                    nn.utils.weight_norm(layer)
                    init.xavier_uniform_(layer.weight)
            except:
                pass


class Conv1D(nn.Module):

    def __init__(self,
                 n_channels,
                 n_kernels,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 last=False,
                 activation=nn.PReLU()):
        super(Conv1D, self).__init__()
        self.conv = nn.Conv1d(
            n_channels, n_kernels,
            kernel_size, stride, padding
        )
        if not last:
            self.net = nn.Sequential(
                self.conv,
                nn.BatchNorm1d(n_kernels),
                activation)
        else:
            self.net = self.conv
        nn.utils.weight_norm(self.conv)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        return self.net(x)


class Deconv1D(nn.Module):

    def __init__(self,
                 n_channels,
                 n_kernels,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 last=False,
                 activation=nn.PReLU()):
        super(Deconv1D, self).__init__()
        self.deconv = nn.ConvTranspose1d(
            n_channels, n_kernels,
            kernel_size, stride, padding
        )
        if not last:
            self.net = nn.Sequential(
                self.deconv,
                nn.BatchNorm1d(n_kernels),
                activation
            )
        else:
            self.net = self.deconv
        nn.init.xavier_uniform_(self.deconv.weight)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):  # 编码层 使用了5层 将1-到128  扩大通道数  卷积核3 步长2 padding 1
    def __init__(self,
                 n_channels=10,  # 1
                 n_kernels=16,  # 128
                 n_layers=3,  # 5
                 seq_size=50):  # 100
        super(Encoder, self).__init__()
        # self.feat_size = (seq_size - 1) // 2 ** n_layers + 1  # 100/32 +1 = 4
        # self.feat_dim = self.feat_size * n_kernels
        self.conv_stack = nn.Sequential(
            *([Conv1D(n_channels, n_kernels // 2 ** (n_layers - 1))] +  # 1, 128/16=8
              [Conv1D(n_kernels // 2 ** (n_layers - l),
                      n_kernels // 2 ** (n_layers - l - 1))
               for l in range(1, n_layers - 1)] +  # for 1到 3 ，那么就是 8,16   16，32  32，64    128/16  128/8 128/4  128/2
              [Conv1D(n_kernels // 2, n_kernels, last=True)])  # 64,128
        )

    def forward(self, x):
        assert len(x.size()) == 3
        feats = self.conv_stack(x)  # 5次卷积下采样 100-50-25-13-7-4
        return feats


class Up(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upsample = Deconv1D(in_ch, in_ch // 2)
        self.conv = Conv1D(in_ch, out_ch)

    def forward(self, x1, x2):  # 64 512 4 # 64 256 7
        x1 = self.upsample(x1)  # 64 256 7
        # Pad x1 to the size of x2
        diff = x2.shape[2] - x1.shape[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetCNN1D(nn.Module):
    # self.unet = UNetBaseline(num_classes=output_size, num_layers=n_layers, features_start=features_start, n_channels=in_size) # 1,5,32,1
    def __init__(
            self,
            num_classes: int = 5,
            num_layers: int = 5,
            features_start: int = 8,  # 32
            n_channels: int = 1
    ):
        super().__init__()
        self.num_layers = num_layers
        layers = [Conv1D(n_channels, features_start)]  # 1，32
        feats = features_start
        for i in range(num_layers - 1):  # 1-3
            layers.append(Conv1D(feats, feats * 2))  # 32,64  #64,128  #128,256
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2))  # 32,64  #64,128  #128,256
            feats //= 2

        conv = nn.Conv1d(feats, num_classes, kernel_size=1)
        conv = nn.utils.weight_norm(conv)
        nn.init.xavier_uniform_(conv.weight)
        layers.append(conv)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        xi = [self.layers[0](x)]

        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))

        for i, layer in enumerate(self.layers[self.num_layers:-1]):  # 5到9
            xi[-1] = layer(xi[-1], xi[-2 - i])

        out = self.layers[-1](xi[-1])
        return out


class CNN1DModel(nn.Module):
    def __init__(self, in_size=1,
                 output_size=5,
                 d_model=128,
                 dropout=0.01,
                 seq_len=9,
                 n_layers=5,
                 pool_filter=16):
        super(CNN1DModel, self).__init__()
        self.enc_net = Encoder(n_channels=in_size, n_kernels=d_model, n_layers=n_layers,
                               seq_size=seq_len)  # 1 128 5 100
        self.pool_filter = pool_filter
        self.mlp_layer = MLPLayer(in_size=d_model * pool_filter, hidden_arch=[1024], output_size=None)
        self.dropout = nn.Dropout(dropout)
        self.pool_filter = pool_filter

        self.fc_out_state = nn.Linear(1024, seq_len)
        nn.init.xavier_normal_(self.fc_out_state.weight)
        self.fc_out_state.bias.data.fill_(0)

    def forward(self, x):
        x = x.unsqueeze(1)
        # x = x.permute(0, 2, 1)  # 64-100-1   转换成64-1-100
        B = x.size(0)  # 64 第一维度
        conv_out = self.dropout(self.enc_net(x))  # 64 128 4
        conv_out = F.adaptive_avg_pool1d(conv_out, self.pool_filter).reshape(x.size(0), -1)  # 64 128 8 后面平铺 64 1024
        mlp_out = self.dropout(self.mlp_layer(conv_out))  # 64 1024
        states_logits = self.fc_out_state(mlp_out) # 64 2 1
        states_logits = states_logits.unsqueeze(-1)
        return states_logits #64 100


class UNETNiLM(nn.Module):
    def __init__(self, in_size=1,
                 output_size=5,  # 1
                 d_model=128,
                 dropout=0.1,
                 seq_len=99,  # 100
                 features_start=32,  # 32
                 n_layers=5,  # 5
                 pool_filter=32):  # 32
        super().__init__()
        self.unet = UNetCNN1D(num_classes=1, num_layers=n_layers, features_start=features_start,
                                 n_channels=in_size)  # 1,5,32,1
        self.conv_layer = Encoder(n_channels=1, n_kernels=d_model, n_layers=n_layers // 2, seq_size=seq_len)
        self.mlp_layer = MLPLayer(in_size=d_model * pool_filter, hidden_arch=[1024], output_size=None)
        self.dropout = nn.Dropout(dropout)
        self.pool_filter = pool_filter

        self.fc_out_state = nn.Linear(1024, seq_len)
        nn.init.xavier_normal_(self.fc_out_state.weight)
        self.fc_out_state.bias.data.fill_(0)

    def forward(self, x):  # unet-encoder-avg_pool1d-
        x = x.unsqueeze(1)
        B = x.size(0)
        # x = x.permute(0, 2, 1)  # 64 1 100
        unet_out = self.dropout(self.unet(x))  # 64 1 25  这是一个标准的unet
        conv_out = self.conv_layer(unet_out)  # 64 128 7  又做了5层下采样
        conv_out = self.dropout(F.adaptive_avg_pool1d(conv_out, self.pool_filter).reshape(x.size(0), -1))  # 64 4096
        mlp_out = self.dropout(self.mlp_layer(conv_out))  # 64 1024
        states_logits = self.fc_out_state(mlp_out)  # 64 2 1
        states_logits = states_logits.unsqueeze(-1)
        return states_logits  # 64 100


