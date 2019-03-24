#!/usr/bin/env python

import json
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from .vectorizer import Vectorizer

class ProjectDataset(Dataset):
  """
    DataSet derived from PyTorch's Dataset class
  """
  def __init__(self, df: pd.DataFrame, vectorizer: Vectorizer=None) -> None:
    self.df = df
    self.df_size = len(self.df)
    self._vectorizer = vectorizer

    # class weights are the inverse of the frequencies of each class
    # this is for cross entropy loss
    # class_counts = df['nationality'].value_counts().to_dict()
    # sorted_counts = sorted(class_counts.items(), key=lambda x:
        # self._vectorizer.nationality_vocab.lookup_token(x[0]))
    # freq = [count for _, count in sorted_counts]
    # self.class_weights = 1.0/torch.tensor(freq, dtype=torch.float32)

  @classmethod
  def load_data_and_create_vectorizer(cls, df: pd.DataFrame):
    """
      Load dataset and create a new Vectorizer object

      Args:
        review_csv: path to the dataset

      Returns:
        an instance of Vectorizer
    """
    return cls(df, Vectorizer.from_dataframe(df))

  @classmethod
  def load_data_and_vectorizer(cls, df: pd.DataFrame, vectorizer_path: str):
    """
      Load dataset and the corresponding vectorizer. Used in the case the
      vectorizer has been cached for re-use

      Args:
        review_csv: path to the dataset
        vectorizer_path: path to the saved vectorizer file
    """
    vectorizer = cls.load_vectorizer(vectorizer_path)
    return cls(df, vectorizer)

  @staticmethod
  def load_vectorizer(vectorizer_path: str) -> Vectorizer:
    """
      A static method for loading the vectorizer from file

      Args:
        vectorizer_path: path to the saved vectorizer file
    """
    with open(vectorizer_path) as f:
      return Vectorizer.from_serializable(json.load(f))

  def save_vectorizer(self, vectorizer_path: str) -> None:
    """
      Saves the vectorizer to disk using json

      Args:
        vectorizer_path: path toe save the vectorizer file
    """
    with open(vectorizer_path, 'w') as f:
      json.dump(self._vectorizer.to_serializeable(), f)

  def get_vectorizer(self) -> Vectorizer:
    return self._vectorizer

  def __len__(self) -> int:
    return self.df_size

  def __getitem__(self, idx: int) -> tuple:
    """
      The primary entry point method for PyTorch datasets

      Args:
        idx: the index to the data point

      Returns:
        a tuple holding the data point's features and label target
    """
    row = self.df.iloc[idx]
    surname_vector = np.asarray(self._vectorizer.vectorize(row['surname']), dtype=np.float32)
    nationality_idx = np.asarray(self._vectorizer.nationality_vocab.lookup_token(
      row['nationality']), dtype=np.float32)
    return (surname_vector, nationality_idx)

  def get_num_batches(self, batch_size: int) -> int:
    """
      Given a batch size, return the number of batches in the dataset
    """
    return len(self) // batch_size
