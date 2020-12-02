#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import sys
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U



class CNN(nn.Module):
    """ Character-based convolutional encoder:
        - 1D Convolution on character-based word embedding followed by non-linear and max pooling functions
        - Character-aware neural language models (Kim et al., 2016)
    """

    def __init__(self, char_embed_size, m_word, kernel_size, num_filters):
        """ Init CNN.

        @param char_embed_size (int): Char embedding size (dimensionality)
        @param m_word (int): Maximum word length (dimensionality)
        @param kernel_size (int): Also known as window size for computing output features (dimensionality)
        @param num_filters (int): Also known as number of output channels (dimensionality)
        
        """
        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size
        self.m_word = m_word
        self.kernel_size = kernel_size
        self.num_filters = num_filters

        # default values
        self.conv1d = None
        self.maxpool2d = None

        ### Initializing the following variables:
        
        self.conv1d = nn.Conv1d(self.char_embed_size, self.num_filters, self.kernel_size)
        
        ### END YOUR CODE

    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        """ Take a mini-batch of x_reshaped tensors and use CNN network to return final output after applying 1D convolutions and pooling.

        @param x_reshaped torch.Tensor: x_emb reshaped as Tensor, with shape (b, e, m), where b = batch size, e = char embed size, m = max word length.
        
        @returns x_conv_out (Tensor): final output of character-based convolutional encoder Tensor, with shape (b, e), where b = batch size, e = embed size.
        """
        
        # Compute 1D convolution
        x_conv = self.conv1d(x_reshaped)
        #print(x_conv.shape)

        # Apply ReLU and perform max pooling
        conv_relu = F.relu(x_conv)
        x_conv_out_values, x_conv_out_idx = torch.max(conv_relu,2)
        #print(x_conv_out_values)
        
        return x_conv_out_values

### END YOUR CODE

