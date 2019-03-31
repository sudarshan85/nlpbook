#!/usr/bin/env python

import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader

class ModelContainer(object):
  def __init__(self, model, optimizer, loss_fn, scheduler=None):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.scheduler = scheduler

class DataContainer(object):
  def __init__(self, df_with_split: pd.DataFrame, dataset_class, vectorizer_file: Path, batch_size:
      int, with_test=True, is_load: bool=True) -> None:
    self.train_df = df_with_split.loc[df_with_split['split'] == 'train']
    self.val_df = df_with_split.loc[df_with_split['split'] == 'val']
    self._bs = batch_size
    self.with_test = with_test

    self.is_load = is_load
    self._lengths = {'train_size': len(self.train_df), 'val_size': len(self.val_df)}
    self._n_batches = [self._lengths['train_size'] // self._bs, self._lengths['val_size'] //
        self._bs]

    if not self.is_load:
      print("Creating and saving vectorizer")
      train_ds = dataset_class.load_data_and_create_vectorizer(self.train_df)
      train_ds.save_vectorizer(vectorizer_file)

    self.train_ds = dataset_class.load_data_and_vectorizer_from_file(self.train_df, vectorizer_file)
    self.vectorizer = self.train_ds.vectorizer
    self.surname_vocab = self.vectorizer.surname_vocab
    self.nationality_vocab = self.vectorizer.nationality_vocab
    self.train_dl = DataLoader(self.train_ds, self._bs, shuffle=True, drop_last=True)

    self.val_ds = dataset_class.load_data_and_vectorizer(self.val_df, self.vectorizer)
    self.val_dl = DataLoader(self.val_ds, self._bs, shuffle=True, drop_last=True)

    if self.with_test:
      self.test_df = df_with_split.loc[df_with_split['split'] == 'test']
      self._lengths['test_size'] = len(self.test_df)
      self._n_batches.append(self._lengths['test_size'] // self._bs)
      self.test_ds = dataset_class.load_data_and_vectorizer(self.test_df, self.vectorizer)
      self.test_dl = DataLoader(self.test_ds, self._bs, shuffle=True, drop_last=True)

  def get_loaders(self):
    return self.train_dl, self.val_dl, self.test_dl

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
  def vocab_size(self):
    return len(self.surname_vocab)

  @property
  def n_classes(self):
    return len(self.nationality_vocab)

  @property
  def sizes(self):
    return self._lengths

