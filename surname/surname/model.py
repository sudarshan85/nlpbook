#!/usr/bin/env python

import torch
from torch import nn

class MLPClassifier(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(MLPClassifier, self).__init__()
    self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
    self.drop = nn.Dropout()
    self.softmax = nn.Softmax()

  def forward(self, x_in, apply_softmax=False):
    y_out = self.fc1(x_in)
    y_out = self.relu(y_out)
    y_out = self.drop(y_out)
    y_out = self.fc2(y_out)

    if apply_softmax:
      y_out = self.softmax(y_out)

    return y_out

class CNNClassifier(nn.Module):
  def __init__(self, initial_num_channels, num_classes, num_channels):
    super(CNNClassifier, self).__init__()

    self.convnet = nn.Sequential(
          nn.Conv1d(in_channels=initial_num_channels, out_channels=num_channels, kernel_size=3),
          nn.ELU(),
          nn.Conv1d(in_channels=initial_num_channels, out_channels=num_channels, kernel_size=3,
            stride=2),
          nn.ELU(),
          nn.Conv1d(in_channels=initial_num_channels, out_channels=num_channels, kernel_size=3,
            stride=2),
          nn.ELU(),
          nn.Conv1d(in_channels=initial_num_channels, out_channels=num_channels, kernel_size=3,
            stride=2),
          nn.ELU()
        )
    self.fc = nn.Linear(num_channels, num_classes)
    self.softmax = nn.Softmax()

  def forward(self, x_in, apply_softmax=False):
    y_out = self.convnet(x_in).squeeze(dim=2)
    y_out = self.fc(y_out)

    if apply_softmax:
      y_out = self.softmax(y_out)

    return y_out
