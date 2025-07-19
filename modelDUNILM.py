import torch
import torch.nn as nn
import torch.nn.functional as F

import  numpy as np
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim = (-d))
        r = torch.stack((t.real, t.imag), -1)
        return r
    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:,:,0], x[:,:,1]), dim = (-d))
        return t.real


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # Vc = torch.fft.rfft(v, 1, onesided=False)
    Vc = rfft(v, 1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


class dct_channel_block(nn.Module):
    def __init__(self, channel ):
        super(dct_channel_block, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1) #innovation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//4, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(channel//4, channel, bias=False),
            nn.Sigmoid()
        )
        # self.dct_norm = nn.LayerNorm([512], eps=1e-6)

        self.dct_norm = nn.LayerNorm([channel], eps=1e-6)  # for lstm on length-wise
        # self.dct_norm = nn.LayerNorm([36], eps=1e-6)#for lstm on length-wise on ill with input =36

    def forward(self, x):
        b, c, l = x.size()  # (B,C,L) (32,96,512)
        # y = self.avg_pool(x) # (B,C,L) -> (B,C,1)

        # y = self.avg_pool(x).view(b, c) # (B,C,L) -> (B,C,1)
        # print("y",y.shape
        # y = self.fc(y).view(b, c, 96)
        list = []
        for i in range(c):
            freq = dct(x[:, i, :])
            # print("freq-shape:",freq.shape)
            list.append(freq)

        stack_dct = torch.stack(list, dim=1)
        # stack_dct = torch.tensor(stack_dct)
        '''
        for traffic mission:f_weight = self.dct_norm(f_weight.permute(0,2,1))#matters for traffic datasets
        '''

        # stack_dct = self.dct_norm(stack_dct)
        lr_weight = self.fc(stack_dct)
        lr_weight = self.dct_norm(lr_weight)

        # print("lr_weight",lr_weight.shape)
        # return x * lr_weight  # result
        return lr_weight
        # return stack_dct  # result


class Inception1D(nn.Module):
    def __init__(self):
        """
        Inception 1D 模块的实现。

        参数:
            in_channels (int): 输入通道数。
            out_channels_1x1 (int): 1x1 卷积的输出通道数。
            out_channels_3x1_reduce (int): 3x1 卷积前的 1x1 降维卷积的输出通道数。
            out_channels_3x1 (int): 3x1 卷积的输出通道数。
            out_channels_5x1_reduce (int): 5x1 卷积前的 1x1 降维卷积的输出通道数。
            out_channels_5x1 (int): 5x1 卷积的输出通道数。
            out_channels_9x1_reduce (int): 9x1 卷积前的 1x1 降维卷积的输出通道数。
            out_channels_9x1 (int): 9x1 卷积的输出通道数。
            out_channels_pool (int): 池化后的 1x1 卷积的输出通道数。
        """
        super(Inception1D, self).__init__()


        # 3x1 卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 5x1 卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU()
        )

        # 9x1 卷积分支
        self.branch4 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.ReLU()
        )
        # 池化分支
        self.branch5 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(1, 16, kernel_size=1),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv1d(64, 16, kernel_size=1),
            nn.ReLU()
        )
    def forward(self, x):
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        branch5 = self.branch5(x)

        # 在通道维度上拼接所有分支的输出
        outputs = torch.cat([branch2, branch3, branch4, branch5], dim=1)
        outputs = self.conv(outputs)
        return outputs





class Conv1DBlockXZ(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_rate=0.5):
        super(Conv1DBlockXZ, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        # self.mamba =Mamba(channel=in_channels, d_model=out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.dropout(x)
        # res = x
        # x = self.conv2(x)
        # x = self.bn(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = x + res
        # x = self.mamba(x)
        # x = self.dropout(x)
        return F.max_pool1d(x, kernel_size=2)
class Deconv1D(nn.Module):

    def __init__(self,
                 n_channels,
                 n_kernels,
                 kernel_size=4,
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

class Deconv1D1(nn.Module):

    def __init__(self,
                 n_channels,
                 n_kernels,
                 kernel_size=2,
                 stride=2,
                 padding=0,
                 last=False,
                 activation=nn.PReLU()):
        super(Deconv1D1, self).__init__()
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
class Conv1D(nn.Module):

    def __init__(self,
                 n_channels,
                 n_kernels,
                 kernel_size=3,
                 stride=1,
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
class Up(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upsample = Deconv1D(in_ch, in_ch // 2)
        self.conv = Conv1D(in_ch, out_ch)

    def forward(self, x1, x2):  # 64 512 4 # 64 256 7
        x1 = self.upsample(x1)  # 64 256 7
        # Pad x1 to the size of x2
        # diff = x2.shape[2] - x1.shape[2]
        # x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up1(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upsample = Deconv1D1(in_ch, in_ch // 2)
        self.conv = Conv1D(in_ch+in_ch // 2, out_ch)

    def forward(self, x1, x2, x3):  # 64 512 4 # 64 256 7
        x1 = self.upsample(x1)  # 64 256 7
        # Pad x1 to the size of x2
        # diff = x2.shape[2] - x1.shape[2]
        # x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x3, x2, x1], dim=1)
        return self.conv(x)


class Deconv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dropout_rate=0.5):
        super(Deconv1DBlock, self).__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.dropout(x)
        return x
class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(UNet1D, self).__init__()
        self.dropout_rate = dropout_rate

        # self.encoder1 = Conv1DBlockXZ(in_channels, 32, kernel_size=9, stride=1, padding=4, dropout_rate=dropout_rate)
        self.encoder1 = Conv1DBlockXZ(16, 32, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate)
        self.encoder2 = Conv1DBlockXZ(32, 64, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate)
        self.encoder3 = Conv1DBlockXZ(64, 128, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate)
        self.encoder4 = Conv1DBlockXZ(128, 256, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate)
        # self.encoder5 = Conv1DBlockXZ(256, 512, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate)


        # self.decoder0 = Up(512, 256)
        self.decoder1 = Up(256, 128)
        self.decoder2 = Up(128, 64)
        self.decoder3 = Up(64, 32)

        self.conv = Deconv1DBlock(32, 16, kernel_size=4, stride=2, padding=1 )


    def forward(self, x, encoder_features=None):
        # 编码器

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        # e5 = self.encoder5(e4)
        # d1 = self.decoder0(e5,e4)

        d2 = self.decoder1(e4,e3)
        d3 = self.decoder2(d2,e2)
        d4 = self.decoder3(d3,e1)
        fina = self.conv(d4)
        # return fina, d1, d2, d3, d4
        return fina,  d2, d3, d4

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(UNet2D, self).__init__()
        self.dropout_rate = dropout_rate
        self.encoder1 = Conv1DBlockXZ(16, 32, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate)
        self.encoder2 = Conv1DBlockXZ(32, 64, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate)
        self.encoder3 = Conv1DBlockXZ(64, 128, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate)
        self.encoder4 = Conv1DBlockXZ(128, 256, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate)
        # self.encoder5 = Conv1DBlockXZ(256, 512, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate)
        #
        #
        # self.decoder0 = Up(512, 256)
        self.decoder1 = Up1(256, 128)
        self.decoder2 = Up1(128, 64)
        self.decoder3 = Up1(64, 32)


    def forward(self, x, d2, d3, d4 ):
        # 编码器
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        # e5 = self.encoder5(e4)
        # dd1 = self.decoder0(e5,e4)

        dd2 = self.decoder1(e4,e3,d2)
        dd3 = self.decoder2(dd2,e2,d3)
        dd4 = self.decoder3(dd3,e1,d4)
        # fina = self.conv(dd4)
        return dd4



class DU_NILM(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(DU_NILM, self).__init__()
        self.unet1 = UNet1D(1, 1, dropout_rate=dropout_rate)
        self.unet2 = UNet2D(1, 1, dropout_rate=dropout_rate)
        self.sigmoid = nn.Sigmoid()  # 添加 Sigmoid 激活函数
        self.conv = Deconv1DBlock(32, 1, kernel_size=3, stride=1, padding=1)
        self.tztq = Inception1D()
        # self.dense = nn.Sequential(
        #     nn.Linear(4810, 1024),  # 输入特征数为64 * conv_output_length
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #
        #     nn.Linear(1024, 599)  # 输出长度与输入序列长度相同
        # )
        self.dropout = nn.Dropout(0.1)
        self.fc_out_state = nn.Linear(1024, 480)
        # self.mlp_layer = MLPLayer(in_size=32* 240, hidden_arch=[1024], output_size=None)
        self.dct = dct_channel_block(480)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.tztq(x)

        # 第一个 U-Net
        u1_out,  d2, d3, d4 = self.unet1(x)  # 获取第一个 U-Net 的输出和编码器特征图

        # 将第一个 U-Net 的输出通过 Sigmoid 激活函数
        attention_mask = self.sigmoid(u1_out)

        # 确保 attention_mask 的尺寸与 x 匹配
        # if attention_mask.shape[-1] != x.shape[-1]:
        #     attention_mask = F.interpolate(attention_mask, size=x.shape[-1], mode='nearest')

        # 将注意力掩码与原始输入相乘

        masked_input = x * attention_mask
        x1 = self.dct(u1_out)
        masked_input = masked_input * x1

        # 第二个 U-Net 的输入是经过掩码处理的输入
        # u2_out = self.unet2(masked_input,  d1, d2, d3, d4)  # 使用第一个 U-Net 的编码器特征图
        u2_out = self.unet2(masked_input,  d2, d3, d4)  # 使用第一个 U-Net 的编码器特征图

        u2_out  = self.conv(u2_out)
        # conv_out = u2_out.reshape(x.size(0), -1)  # 64 4096
        # mlp_out = self.dropout(self.mlp_layer(conv_out))  # 64 1024
        # states_logits = self.fc_out_state(mlp_out)  # 64 2 1
        # states_logits = states_logits.unsqueeze(-1)
        u2_out = u2_out.transpose(1, 2)
        return u2_out  # 64 100





