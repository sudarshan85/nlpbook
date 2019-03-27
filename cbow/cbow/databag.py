#!/usr/bin/env python

import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from .dataset import CBOWDataset

class DataBag(object):
  def __init__(self, df_with_split: pd.DataFrame, vectorizer_file: Path, batch_size: int, is_load:
      bool=True) -> None:
    self._train_df = df_with_split.loc[df_with_split['split'] == 'train']
    self._val_df = df_with_split.loc[df_with_split['split'] == 'val']
    self._test_df = df_with_split.loc[df_with_split['split'] == 'test']
    self.bs = batch_size
    self.is_load = is_load

    self._lengths = (len(self._train_df), len(self._val_df), len(self._test_df))

    if not self.is_load:
      print("Creating and saving vectorizer")
      train_ds = CBOWDataset.load_data_and_create_vectorizer(self._train_df)
      train_ds.save_vectorizer(vectorizer_file)

    self.train_ds = CBOWDataset.load_data_and_vectorizer_from_file(self._train_df, vectorizer_file)
    self._vectorizer = self.train_ds.get_vectorizer()
    self.train_dl = DataLoader(self.train_ds, batch_size, shuffle=True, drop_last=True)

    self.val_ds = CBOWDataset.load_data_and_vectorizer(self._val_df, vectorizer_file)
    self.val_dl = DataLoader(self.val_ds, batch_size, shuffle=True, drop_last=True)

    self.test_ds = CBOWDataset.load_data_and_vectorizer(self._test_df, vectorizer_file)
    self.test_dl = DataLoader(self.test_ds, batch_size, shuffle=True, drop_last=True)

  def get_vectorizer(self):
    return self._vectorizer

  def get_loaders(self):
    return self.train_dl, self.val_dl, self.test_dl

  def __repr__(self):
    r = []
    r.append(f'Training set size {self._lengths[0]}\n')
    r.append(f'Validation set size {self._lengths[1]}\n')
    r.append(f'Testing set size {self._lengths[2]}\n')
    r.append(f'Batch size {self.bs}')
    return ''.join(r)

  @property
  def len(self):
    return self._lengths

