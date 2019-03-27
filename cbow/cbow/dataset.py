#!/usr/bin/env python

import pandas as pd
import numpy as np
import json

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from .vectorizer import Vectorizer

class CBOWDataset(Dataset):
  def __init__(self, df: pd.DataFrame, vectorizer: Vectorizer) -> None:
    self._df = df
    self._vectorizer = vectorizer

    # get the length of the longest context
    # should be window_size * 2 and target is 1
    measure_len = lambda context: len(context.split(' '))
    self._max_seq_len = max(map(measure_len, self._df['context']))

  @classmethod
  def load_data_and_create_vectorizer(cls, df: pd.DataFrame):
    return cls(df, Vectorizer.from_dataframe(df))

  @staticmethod
  def load_vectorizer(vectorizer_path: Path) -> Vectorizer:
    with open(vectorizer_path) as fp:
      return Vectorizer.from_serializable(json.load(fp))

  @classmethod
  def load_data_and_vectorizer_from_file(cls, df: pd.DataFrame, vectorizer_path: Path):
    vectorizer = cls.load_vectorizer(vectorizer_path)
    return cls(df, vectorizer)

  @classmethod
  def load_data_and_vectorizer(cls, df: pd.DataFrame, vectorizer: Vectorizer):
    return cls(df, vectorizer)

  def save_vectorizer(self, vectorizer_path: Path) -> None:
    with open(vectorizer_path, 'w') as fp:
      json.dump(self._vectorizer.to_serializable(), fp)

  def get_vectorizer(self):
    return self._vectorizer

  def __getitem__(self, idx):
    row = self._df.iloc[idx]
    context_vector = np.asarray(self._vectorizer.vectorizer(row['context'], self._max_seq_len))
    target_idx = np.asarray(self._vectorizer.cbow_vocab.lookup_token(row['target']))

    return (context_vector, target_idx)

  def __len__(self):
    return len(self._df)

  @property
  def max_length(self):
    return self._max_seq_len

  def get_num_batches(self, batch_size: int) -> int:
    return len(self) // batch_size

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
    self._vocab_size = len(self._vectorizer.cbow_vocab)
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
  def vocabulary_size(self):
    return self._vocab_size

  @property
  def sizes(self):
    return self._lengths

