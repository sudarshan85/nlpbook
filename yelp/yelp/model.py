#!/usr/bin/env python

import torch
from torch import nn

class Classifier(nn.Module):
  def __init__(self, num_features):
    super(Classifier, self).__init__()
    self.fc1 = nn.Linear(in_features=num_features, out_features=1)

  def forward(self, x_in, apply_sigmoid=False):
    y_out = self.fc1(x_in).squeeze(1)
    if apply_sigmoid:
      y_out = torch.sigmoid(y_out)

    return y_out
