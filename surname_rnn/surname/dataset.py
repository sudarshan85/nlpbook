#!/usr/bin/env python

import pandas as pd
import numpy as np
import json
import torch

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from .vectorizer import Vectorizer

class SurnameDataset(Dataset):
  def __init__(self, df: pd.DataFrame, vectorizer: Vectorizer) -> None:
    self.df = df
    self.vectorizer = vectorizer

    # +2 to account for EOS and BOS
    self.max_seq_len = max(map(len, self.df['surname'])) + 2

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
      json.dump(self.vectorizer.to_serializable(), fp)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    surname_vector, vec_length = np.asarray(self.vectorizer.vectorize(row['surname'],
      self.max_seq_len))
    nationality_idx = np.asarray(self.vectorizer.nationality_vocab.lookup_token(row['nationality']))

    return (surname_vector, vec_length, nationality_idx)

  def __len__(self):
    return len(self.df)

