#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, word_size):
        super(Highway, self).__init__()
        self.w_proj = nn.Linear(word_size, word_size)
        self.w_gate = nn.Linear(word_size, word_size)
        self.dim_size = word_size

    def forward(self, inp):
        gate = torch.sigmoid(self.w_gate(inp))
        return F.relu(self.w_proj(inp)) * gate + (1. - gate) * inp
### END YOUR CODE 

