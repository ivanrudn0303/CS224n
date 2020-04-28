#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, inp_dim, out_dim, padding=1, window=5):
        super(CNN, self).__init__()
        self.cnn = nn.Conv1d(inp_dim, out_dim, window, padding=padding)

    def forward(self, inp):
        return F.relu(self.cnn(inp)).max(dim=-1)
### END YOUR CODE

