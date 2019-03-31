#!/usr/bin/env python

import torch
from torch import nn
from typing import Tuple

from .elman import ElmanRNN

def column_gather(y_out: torch.FloatTensor, x_lens: torch.FloatTensor) -> torch.FloatTensor:
  """
    Get a specific vector from each batch datapoint in 'y_out'
    Iteratove over batch row indices, get the vector thats at the position
    indicated by the corresponding value in 'x_lens' at the row index

    Args:
      y_out: shape (bs, seq_sz, feat_sz)
      x_lens: shape (bs,)

    Returns:
      y_out: shape (bs, feat_sz)
  """
  x_lens = x_lens.long().detach().cpu().numpy()-1

  out = []
  for batch_idx, column_idx in enumerate(x_lens):
    out.append(y_out[batch_idx, column_idx])

  return torch.stack(out)

class SurnameClassifier(nn.Module):
  """
    A Classifier with a RNN to extract features and an MLP to classify
  """
  def __init__(self, emb_sz: int, n_embs: int, n_classes: int, rnn_hidden_sz:int ,
               batch_first: bool=True, padding_idx: int=0) -> None:
    """
      Args:
        emb_sz: the size of the character embeddings
        n_embs: the number of characters to embed (vocabulary size)
        n_classes: the size of the prediction vector
        rnn_hidden_sz: the size of RNN's hidden state
        batch_first: informs wehther the input tensors will have batch or sequence on the 0th dim
        padding_idx: idx for the tensor padding
    """
    super(SurnameClassifier, self).__init__()
    self.emb = nn.Embedding(n_embs, emb_sz, padding_idx)
    self.rnn = ElmanRNN(inp_sz=emb_sz, hidden_sz=rnn_hidden_sz, batch_first=batch_first)
    self.dropout = nn.Dropout(0.5)
    self.mlp = nn.Sequential(
      nn.Linear(rnn_hidden_sz, rnn_hidden_sz),
      nn.ReLU(),
      self.dropout,
      nn.Linear(rnn_hidden_sz, n_classes)
    )
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x_in: torch.Tensor, x_lens: torch.Tensor=None,
      apply_softmax: bool=False) -> torch.Tensor:
    """
      The forward pass of the classifier

      Args:
        model_inp: A tuple contain the input tensor and the lengths of each sequence in the batch
        used to find the final vector of each sequence
        apply_softmax: flag for softmax activation, should be false when used with nn.CrossEntropy
    """
    x_emb = self.emb(x_in)
    y_out = self.rnn(x_emb)

    if x_lens is not None:
      y_out = column_gather(y_out, x_lens)
    else:
      # since batch_first is true, the output of ElmanRNN is of shape (bs, seq_sz, hidden_sz)
      # this grabs the last hidden vector of each sequence of each batch
      # so y_out shape goes from (bs, seq_sz, feat_sz) to (bs, feat_sz)
      y_out = y_out[:, -1, :]

    y_out = self.dropout(y_out)
    y_out = self.mlp(y_out)

    if apply_softmax:
      y_out = self.softmax(y_out)

    return y_out
