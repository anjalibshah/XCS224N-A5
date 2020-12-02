#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

# ----------
# CONSTANTS
# ----------
BATCH_SIZE = 5
CHAR_EMBED_SIZE = 50
KERNEL_SIZE = 5
DROPOUT_RATE = 0.3
M_WORD = 21

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f

        self.embed_size = embed_size
        self.CHAR_EMBED_SIZE = CHAR_EMBED_SIZE
        self.vocab = vocab
        # default values
        self.embeddings = None
        self.cnn = None
        self.highway = None
        
        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), self.CHAR_EMBED_SIZE, pad_token_idx)
        self.cnn = CNN(CHAR_EMBED_SIZE, M_WORD, KERNEL_SIZE, self.embed_size)
        self.highway = Highway(self.embed_size, DROPOUT_RATE)
        
        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        combined_outputs = []
        

        for s in torch.split(input_tensor, 1):
            #print(s.shape)
            x_padded = torch.squeeze(s, dim=0)
            x_emb = self.embeddings.forward(x_padded)
            #print(x_emb.shape)
            x_conv_out = self.cnn.forward(x_emb.permute(0,2,1))
            x_word_emb = self.highway.forward(x_conv_out)
            combined_outputs.append(x_word_emb)

        return torch.stack(combined_outputs)

        ### END YOUR CODE
