#!/usr/bin/env python

import torch
from torch import nn

class ElmanRNN(nn.Module):
  """
    An Elman RNN built using RNNCell
  """
  def __init__(self, inp_sz: int, hidden_sz: int, batch_first: bool=False) -> None:
    """
      Args:
        inp_sz: size of input vectors
        hidden_sz: size of hidden state vectors
        batch_first: whether the 0th dim is batch
    """
    super(ElmanRNN, self).__init__()
    self.batch_first = batch_first
    self.hidden_sz = hidden_sz
    self.rnn_cell = nn.RNNCell(inp_sz, hidden_sz)

  def _init_hidden(self, bs: int) -> torch.Tensor:
    return torch.zeros((bs, self.hidden_sz))

  def forward(self, x_in: torch.Tensor, init_hidden: torch.Tensor=None) -> torch.Tensor:
    """
      The forward pass of ElmanRNN

      Args:
        x_in: an input data tensor
          If self.batch_first: x_in.shape = (bs, seq_sz, feat_sz)
          Else: x_in.shape = (seq_sz, bs, feat_sz)

      Returns:
        hiddens: the outputs of the RNN at each time step
          If self.batch_first: hiddens.shape = (bs, seq_sz, hidden_sz)
          Else: x_in.shape = (seq_sz, bs, hidden_sz)
    """
    bs, seq_sz, feat_sz = x_in.size()
    if self.batch_first:
      x_in = x_in.permute(1, 0, 2)

    hiddens = []
    if init_hidden is None:
      # remember .to so that newly initialized tensors are placed on the
      # same device as other tensors
      init_hidden = self._init_hidden(bs).to(x_in.device)

    hidden_t = init_hidden

    for t in range(seq_sz):
      hidden_t = self.rnn_cell(x_in[t], hidden_t)
      hiddens.append(hidden_t)

    hiddens = torch.stack(hiddens)

    if self.batch_first:
      hiddens = hiddens.permute(1, 0, 2)

    return hiddens
