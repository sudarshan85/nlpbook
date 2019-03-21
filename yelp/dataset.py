#!/usr/bin/env python

import json
import pandas as pd

from torch.utils.data import Dataset

from .vectorizer import Vectorizer

class ProjectDataset(Dataset):
  """
    DataSet derived from PyTorch's Dataset class
  """
  def __init__(self, review_df: pd.DataFrame, vectorizer: Vectorizer) -> None:
    self._review_df = review_df
    self._vectorizer = vectorizer

    self._train_df = self._review_df[self._review_df['split'] == 'train']
    self._train_size = len(self._train_df)

    self._val_df = self._review_df[self._review_df['split'] == 'val']
    self._val_size = len(self._val_df)

    self._test_df = self._review_df[self._review_df['split'] == 'test']
    self._test_size = len(self._test_df)

    self._lookup_dict = {
          'train': (self._train_df, self._train_size),
          'val': (self._val_df, self._val_size),
          'test': (self._test_df, self._test_size)
        }

    self.set_split('train')

  def set_split(self, split: str='train') -> None:
    """
      Selects the splits in the dataset using the split column in the dataframe

      Args:
        split: one of 'train', 'val', 'test'
    """
    self._target_split = split
    self._target_df, self._target_size = self._lookup_dict[split]

  @classmethod
  def load_data_and_create_vectorizer(cls, review_csv: str):
    """
      Load dataset and create a new Vectorizer object

      Args:
        review_csv: path to the dataset

      Returns:
        an instance of Vectorizer
    """

    review_df = pd.read_csv(review_csv)
    train_df = review_df[review_df['split'] == 'train']
    return cls(review_df, Vectorizer.from_dataframe(train_df))


  @classmethod
  def load_data_and_vectorizer(cls, review_csv: str, vectorizer_path: str):
    """
      Load dataset and the corresponding vectorizer. Used in the case the
      vectorizer has been cached for re-use

      Args:
        review_csv: path to the dataset
        vectorizer_path: path to the saved vectorizer file
    """
    review_df = pd.read_csv(review_csv)
    vectorizer = cls.load_vectorizer(vectorizer_path)
    return cls(review_df, vectorizer)

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
    return self._target_size

  def __getitem__(self, idx: int) -> dict:
    """
      The primary entry point method for PyTorch datasets

      Args:
        idx: the index to the data point

      Returns:
        a dictionary holding the data point's features (x_data) and label
        (y_target)
    """
    row = self._target_df.iloc[idx]
    review_vector = self._vectorizer.vectorize(row['review'])
    rating_idx = self._vectorizer.rating_vocab.lookup_token(row['rating'])

    return {'x_data': review_vector, 'y_target': rating_idx}

  def get_num_batches(self, batch_size: int) -> int:
    """
      Given a batch size, return the number of batches in the dataset
    """
    return len(self) // batch_size
