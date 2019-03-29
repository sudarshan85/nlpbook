#!/usr/bin/env python

import pdb
import torch
from torch import nn
from torch.nn import functional as F

class NewsClassifier(nn.Module):
  def __init__(self, emb_sz, vocab_size, n_channels, hidden_dim, n_classes, dropout_p,
      pretrained=None, freeze_pretrained=False, padding_idx=0):
    super(NewsClassifier, self).__init__()

    if pretrained is None:
      self.emb = nn.Embedding(vocab_size, emb_sz, padding_idx)
    else:
      pretrained_emb = torch.from_numpy(pretrained).float()
      self.emb = nn.Embedding(vocab_size, emb_sz, padding_idx, _weight=pretrained_emb)
      if freeze_pretrained:
        self.emb.weight.requires_grad = False

    self.convnet = nn.Sequential(
      nn.Conv1d(in_channels=emb_sz, out_channels=n_channels, kernel_size=3),
      nn.ELU(),
      nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=2),
      nn.ELU(),
      nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=2),
      nn.ELU(),
      nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=3),
      nn.ELU()
    )

    self.dropout = nn.Dropout(p=dropout_p)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(in_features=n_channels, out_features=hidden_dim)
    self.fc2 = nn.Linear(in_features=hidden_dim, out_features=n_classes)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x_in, apply_softmax=False):
    # embed and permute so features are channels
    # conv1d (batch, channels, input)
    x_emb = self.emb(x_in).permute(0,2,1)
    features = self.convnet(x_emb)

    # average and remove extra dimension
    remaining_size = features.size(dim=2)
    features = F.avg_pool1d(features, remaining_size).squeeze(dim=2)
    features = self.dropout(features)

    # mlp classifier
    hidden_vector = self.fc1(features)
    hidden_vector = self.dropout(hidden_vector)
    hidden_vector = self.relu(hidden_vector)
    prediction_vector = self.fc2(hidden_vector)

    if apply_softmax:
      prediction_vector = self.softmax(prediction_vector)

    return prediction_vector

class ModelContainer(object):
  def __init__(self, model, optimizer, loss_fn, scheduler=None):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.scheduler = scheduler
