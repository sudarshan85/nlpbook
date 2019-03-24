#!/usr/bin/env python

import pandas as pd
import numpy as np
import string

from .vocab import Vocabulary

class Vectorizer(object):
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
      one_hot[self.surname_vocab.lookup_token(token)] = 1

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


