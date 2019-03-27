#!/usr/bin/env python

from torch import nn

class CBOWClassifier(nn.Module):
  def __init__(self, vocabulary_size, embedding_size, padding_idx=0):
    super(CBOWClassifier, self).__init__()
    self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_size,
        padding_idx=padding_idx)
    self.dropout = nn.Dropout(p=0.3)
    self.fc1 = nn.Linear(in_features=embedding_size, out_features=vocabulary_size)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x_in, apply_softmax=False):
    x_emb_sum = self.dropout(self.embedding(x_in).sum(dim=1))
    y_out = self.fc1(x_emb_sum)

    if apply_softmax:
      y_out = self.softmax(y_out)

    return y_out

class ModelContainer(object):
  def __init__(self, model, optimizer, loss_fn, scheduler=None):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.scheduler = scheduler
