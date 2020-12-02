#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U



class Highway(nn.Module):
    """ Simple Highway Block:
        - Skip connection controlled by a dynamic gate
        - Highway Networks (Srivastava, et al. 2015)
    """

    def __init__(self, embed_size, dropout_rate):
        """ Init Highway Network.

        @param embed_size (int): Embedding size (dimensionality)
        @param dropout_rate (float): Dropout probability
        """
        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.dropout_rate = dropout_rate
        
        # default values
        self.w_projection = None
        self.w_gate = None
        self.dropout = None

        ### Initializing the following variables:
        
        self.w_projection = nn.Linear(self.embed_size, self.embed_size)
        self.w_gate = nn.Linear(self.embed_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        ### END YOUR CODE

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """ Take a mini-batch of x_conv_out tensors and used the highway network architecture to return a character-based word embedding.

        @param x_conv_out torch.Tensor: final output of character-based convolutional encoder Tensor, with shape (b, e), where b = batch size, h = embed size.
        
        @returns x_word_emb (Tensor): a character-based input word embedding Tensor, with shape (e), where e = embed size.
        """
        
        # Compute X projection 
        x_proj = F.relu(self.w_projection(x_conv_out))
        #print(x_proj.shape)

        # Compute X Gate
        x_gate = torch.sigmoid(self.w_gate(x_conv_out))
        #print(x_gate.shape)

        # Compute X highway by using gate to combine the projection with skip connection
        x_highway = torch.mul(x_gate,x_proj) + torch.mul((1-x_gate),x_conv_out)
        #print(x_highway.shape)

        # Apply dropout to get the character-based word embedding
        x_word_emb = self.dropout(x_highway)
        
        return x_word_emb

### END YOUR CODE 

