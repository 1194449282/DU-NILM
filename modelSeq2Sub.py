import math
import torch
from torch import nn
import torch.nn.functional as F

class seq2Subcnn_Pytorch(nn.Module):
    def __init__(self, sequence_length):
        # Refer to "ZHANG C, ZHONG M, WANG Z, et al. Sequence-to-point learning with neural networks for non-intrusive load monitoring[C].The 32nd AAAI Conference on Artificial Intelligence"
        super(seq2Subcnn_Pytorch, self).__init__()
        self.seq_length = sequence_length

        self.conv = nn.Sequential(
            nn.ConstantPad1d((4, 5), 0),
            nn.Conv1d(1, 30, 10, stride=1),
            nn.ReLU(True),
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(30, 30, 8, stride=1),
            nn.ReLU(True),
            nn.ConstantPad1d((2, 3), 0),
            nn.Conv1d(30, 40, 6, stride=1),
            nn.ReLU(True),
            nn.ConstantPad1d((2, 2), 0),
            nn.Conv1d(40, 50, 5, stride=1),
            nn.ReLU(True),
            nn.ConstantPad1d((2, 2), 0),
            nn.Conv1d(50, 50, 5, stride=1),
            nn.ReLU(True)
        )

        self.dense = nn.Sequential(
            nn.Linear(50 * sequence_length, 1024),
            nn.ReLU(),
            nn.Linear(1024, 120)
        )

    def forward(self, x):  # 64 480
        x = x.unsqueeze(1)  # 64 1 480
        x = self.conv(x)  # 64 50 480
        x = self.dense(x.view(-1, 50 * self.seq_length))  # 64 60
        # x = x.view(-1, 1)
        # x = x.view(-1, self.seq_length) #64 480
        x = x.unsqueeze(-1)
        return x