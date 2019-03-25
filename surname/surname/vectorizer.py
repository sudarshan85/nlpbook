#!/usr/bin/env python

import pandas as pd
import numpy as np
import string

from .vocab import Vocabulary

class CNNVectorizer(object):
  """
    The Vectorizer which coordinates the vocabularies and puts them to use
  """
  def __init__(self, surname_vocab: Vocabulary, nationality_vocab: Vocabulary, max_surname_length:
      int) -> None:
    """
      Args:
        surname_vocab     : maps characters to integers
        nationality_vocab : maps nationalities to integers
        max_surname_length: the length of the longest surname
    """
    self.surname_vocab = surname_vocab
    self.nationality_vocab = nationality_vocab
    self.max_surname_length = max_surname_length


  def vectorize(self, surname: str) -> np.ndarray:
    """
      Create a collapsed one-hot vector for the review

      Args:
        review: the review

      Returns:
        one_hot_matrix: a matrix of one-hot vectors
    """
    one_hot = np.zeros((len(self.surname_vocab), self.max_surname_length), dtype=np.float32)

    for idx, char in enumerate(surname):
      char_idx = self.surname_vocab.lookup_token(char)
      one_hot[char_idx][idx] = 1

    return one_hot_matrix


  @classmethod
  def from_dataframe(cls, surname_df: pd.DataFrame):
    """
      Instantiates a Vectorizer object from the dataset dataframe

      Args:
        review_df: the review dataset
        cutoff   : the parameter for frequency-based filtering

      Returns:
        an instance of the Vectorizer
    """
    surname_vocab = Vocabulary(unk_token='@')
    nationality_vocab = Vocabulary(add_unk=False)
    max_surname_length = 0

    for idx, row in surname_df.iterrows():
      max_surname_length = max(max_surname_length, len(row['surname']))
      for char in row['surname']:
        surname_vocab.add_token(char)
      nationality_vocab.add_token(row['nationality'])

    return cls(surname_vocab, nationality_vocab, max_surname_length)

  @classmethod
  def from_serializable(cls, contents: dict):
    """
      Instantiates a Vectorizer from a serializable dictionary
    """
    surname_vocab = Vocabulary.from_serializable(contents['surname_vocab'])
    nationality_vocab = Vocabulary.from_serializable(contents['nationality_vocab'])
    max_surname_length = contents['max_surname_length']

    return cls(surname_vocab=surname_vocab, nationality_vocab=nationality_vocab,
        max_surname_length=max_surname_length)

  def to_serializeable(self) -> dict:
    """
      Create the serializable dictionary for caching
    """
    return {
        'surname_vocab': self.surname_vocab.to_serializeable(),
        'nationality_vocab': self.nationality_vocab.to_serializeable(),
        'max_surname_length': self.max_surname_length
        }

class MLPVectorizer(object):
  """
    The Vectorizer which coordinates the vocabularies and puts them to use
  """
  def __init__(self, surname_vocab: Vocabulary, nationality_vocab: Vocabulary) -> None:
    """
      Args:
        surname_vocab    : maps characters to integers
        nationality_vocab: maps nationalities to integers
    """
    self.surname_vocab = surname_vocab
    self.nationality_vocab = nationality_vocab


  def vectorize(self, surname: str) -> np.ndarray:
    """
      Create a collapsed one-hot vector for the review

      Args:
        review: the review

      Returns:
        one_hot: the collapsed one-hot encoding
    """
    one_hot = np.zeros(len(self.surname_vocab), dtype=np.float32)

    for token in surname:
      one_hot[self.surname_vocab.lookup_token(token)] += 1

    return one_hot


  @classmethod
  def from_dataframe(cls, surname_df: pd.DataFrame):
    """
      Instantiates a Vectorizer object from the dataset dataframe

      Args:
        review_df: the review dataset
        cutoff   : the parameter for frequency-based filtering

      Returns:
        an instance of the Vectorizer
    """
    surname_vocab = Vocabulary(unk_token='@')
    nationality_vocab = Vocabulary(add_unk=False)

    for idx, row in surname_df.iterrows():
      for c in row['surname']:
        surname_vocab.add_token(c)
      nationality_vocab.add_token(row['nationality'])

    return cls(surname_vocab, nationality_vocab)

  @classmethod
  def from_serializable(cls, contents: dict):
    """
      Instantiates a Vectorizer from a serializable dictionary
    """
    surname_vocab = Vocabulary.from_serializable(contents['surname_vocab'])
    nationality_vocab = Vocabulary.from_serializable(contents['nationality_vocab'])

    return cls(surname_vocab=surname_vocab, nationality_vocab=nationality_vocab)

  def to_serializeable(self) -> dict:
    """
      Create the serializable dictionary for caching
    """
    return {
        'surname_vocab': self.surname_vocab.to_serializeable(),
        'nationality_vocab': self.nationality_vocab.to_serializeable()
        }


