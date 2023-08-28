import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim

class ConvolutionBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ConvolutionBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.block(x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, num_heads, dropout, num_conv_blocks):
        super(TransformerClassifier, self).__init__()
        self.conv1 = ConvolutionBlock(input_size, hidden_size)
        self.conv_blocks = nn.ModuleList([
            ConvolutionBlock(hidden_size, hidden_size) for _ in range(num_conv_blocks)
        ])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size, dropout=dropout),
            num_layers
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, hidden_size, seq_len)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Average the sequence dimension
        x = torch.sigmoid(self.fc(x))
        return x