#!/usr/bin/env python

import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from .dataset import CBOWDataset

class DataContainer(object):
  def __init__(self, df_with_split: pd.DataFrame, vectorizer_file: Path, batch_size: int, with_test
      = True, is_load: bool=True) -> None:
    self._train_df = df_with_split.loc[df_with_split['split'] == 'train']
    self._val_df = df_with_split.loc[df_with_split['split'] == 'val']
    self._bs = batch_size

    self.is_load = is_load
    self._lengths = {'train_size': len(self._train_df), 'val_size': len(self._val_df)}

    if not self.is_load:
      print("Creating and saving vectorizer")
      train_ds = CBOWDataset.load_data_and_create_vectorizer(self._train_df)
      train_ds.save_vectorizer(vectorizer_file)

    self._train_ds = CBOWDataset.load_data_and_vectorizer_from_file(self._train_df, vectorizer_file)
    self._vectorizer = self._train_ds.get_vectorizer()
    self.train_dl = DataLoader(self._train_ds, batch_size, shuffle=True, drop_last=True)

    self._val_ds = CBOWDataset.load_data_and_vectorizer(self._val_df, self._vectorizer)
    self.val_dl = DataLoader(self._val_ds, batch_size, shuffle=True, drop_last=True)

    if with_test:
      self._test_df = df_with_split.loc[df_with_split['split'] == 'test']
      self._lengths['test_size'] = len(self._test_df)
      self._test_ds = CBOWDataset.load_data_and_vectorizer(self._test_df, self._vectorizer)
      self.test_dl = DataLoader(self._test_ds, batch_size, shuffle=True, drop_last=True)

  def get_vectorizer(self):
    return self._vectorizer

  def get_loaders(self):
    return self.train_dl, self.val_dl, self.test_dl

  @property
  def sizes(self):
    return self._lengths

