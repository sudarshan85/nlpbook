#!/usr/bin/env python

import pandas as pd
import numpy as np
import json

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from .vectorizer import Vectorizer

class NewsDataset(Dataset):
  def __init__(self, df: pd.DataFrame, vectorizer: Vectorizer) -> None:
    self._df = df
    self._vectorizer = vectorizer

    # +2 to account for EOS and BOS
    measure_len = lambda context: len(context.split(' '))
    self._max_seq_len = max(map(measure_len, self._df['title'])) + 2

  @classmethod
  def load_data_and_create_vectorizer(cls, df: pd.DataFrame):
    return cls(df, Vectorizer.from_dataframe(df))

  @classmethod
  def load_data_and_vectorizer_from_file(cls, df: pd.DataFrame, vectorizer_path: Path):
    vectorizer = cls.load_vectorizer(vectorizer_path)
    return cls(df, vectorizer)

  @classmethod
  def load_data_and_vectorizer(cls, df: pd.DataFrame, vectorizer: Vectorizer):
    return cls(df, vectorizer)

  @staticmethod
  def load_vectorizer(vectorizer_path: Path) -> Vectorizer:
    with open(vectorizer_path) as fp:
      return Vectorizer.from_serializable(json.load(fp))

  def save_vectorizer(self, vectorizer_path: Path) -> None:
    with open(vectorizer_path, 'w') as fp:
      json.dump(self._vectorizer.to_serializable(), fp)

  def __getitem__(self, idx):
    row = self._df.iloc[idx]
    title_vector = np.asarray(self._vectorizer.vectorizer(row['title'], self._max_seq_len))
    category_idx = np.asarray(self._vectorizer.category_vocab.lookup_token(row['category']))

    return (title_vector, category_idx)

  @property
  def vectorizer(self):
    return self._vectorizer

  @property
  def max_seq_length(self):
    return self._max_seq_len

  def __len__(self):
    return len(self._df)

class DataContainer(object):
  def __init__(self, df_with_split: pd.DataFrame, dataset_class, vectorizer_file: Path, batch_size: int, with_test = True, is_load: bool=True) -> None:
    self._train_df = df_with_split.loc[df_with_split['split'] == 'train']
    self._val_df = df_with_split.loc[df_with_split['split'] == 'val']
    self._bs = batch_size
    self.with_test = with_test

    self.is_load = is_load
    self._lengths = {'train_size': len(self._train_df), 'val_size': len(self._val_df)}
    self._n_batches = [self._lengths['train_size'] // self._bs, self._lengths['val_size'] //
        self._bs]

    if not self.is_load:
      print("Creating and saving vectorizer")
      train_ds = dataset_class.load_data_and_create_vectorizer(self._train_df)
      train_ds.save_vectorizer(vectorizer_file)

    self._train_ds = dataset_class.load_data_and_vectorizer_from_file(self._train_df, vectorizer_file)
    self._vectorizer = self._train_ds.vectorizer
    self._vocabulary = self._vectorizer.title_vocab
    self._vocab_size = len(self._vocabulary)
    self._n_classes = len(self._vectorizer.category_vocab)
    self.train_dl = DataLoader(self._train_ds, batch_size, shuffle=True, drop_last=True)

    self._val_ds = dataset_class.load_data_and_vectorizer(self._val_df, self._vectorizer)
    self.val_dl = DataLoader(self._val_ds, batch_size, shuffle=True, drop_last=True)

    if self.with_test:
      self._test_df = df_with_split.loc[df_with_split['split'] == 'test']
      self._lengths['test_size'] = len(self._test_df)
      self._n_batches.append(self._lengths['test_size'] // self._bs)
      self._test_ds = dataset_class.load_data_and_vectorizer(self._test_df, self._vectorizer)
      self.test_dl = DataLoader(self._test_ds, batch_size, shuffle=True, drop_last=True)

  def get_loaders(self):
    return self.train_dl, self.val_dl, self.test_dl

  @property
  def vectorizer(self):
    return self._vectorizer

  @property
  def vocabulary(self):
    return self._vocabulary

  @property
  def train_batches(self):
    return self._n_batches[0]

  @property
  def val_batches(self):
    return self._n_batches[1]

  @property
  def test_batches(self):
    if not self.with_test:
      raise NameError("No test dataset was provided")
    return self._n_batches[2]

  @property
  def vocabulary_size(self):
    return self._vocab_size

  @property
  def n_classes(self):
    return self._n_classes

  @property
  def sizes(self):
    return self._lengths

