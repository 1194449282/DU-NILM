import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """
    Causal Convolution Layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        # Causal convolution: remove the padding from the end
        x = self.conv(x)
        return x[:, :, :-self.padding]

class TemporalBlock(nn.Module):
    """
    Temporal Block with Causal Convolution and Gated Activation.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation=dilation)
        self.net = nn.Sequential(self.conv1, self.relu, self.dropout, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        xx = out + res
        xx = self.relu(xx)
        return xx

class TCN(nn.Module):
    """
    Temporal Convolutional Network (TCN).
    """
    def __init__(self, num_inputs=1, num_channels=[30, 40, 50], kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)
        # Add a final convolution layer to convert to 1 channel
        self.final_conv = nn.Conv1d(num_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.network(x)
        x = self.final_conv(x)
        return x.permute(0, 2, 1)  # Change shape from (64, 1, 480) to (64, 480, 1)

