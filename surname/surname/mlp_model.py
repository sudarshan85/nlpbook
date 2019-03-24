#!/usr/bin/env python

import torch
from torch import nn

class Classifier(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(Classifier, self).__init__()
    self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
    self.softmax = nn.Softmax()

  def forward(self, x_in, apply_softmax=False):
    y_out = self.fc1(x_in)
    y_out = self.relu(y_out)
    y_out = self.fc2(y_out)

    if apply_softmax:
      y_out = self.softmax(y_out)

    return y_out
