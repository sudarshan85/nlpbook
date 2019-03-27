#!/usr/bin/env python

import pandas as pd
import numpy as np

from vocabulary import Vocabulary

class Vectorizer(object):
  def __init__(self, cbow_vocab: Vocabulary):
    self.cbow_vocab = cbow_vocab

  def vectorizer(self, context: str, vector_len: int=-1) -> np.ndarray:
    """
      Args:
        context: string of words separated by a space
        vector_len: an argument for forcing the length of index vector
    """
    idxs = [self.cbow_vocab.lookup_token(token) for token in context.split(' ')]
    if vector_len < 0:
      vector_len = len(idxs)

    out_vector = np.zeros(vector_len, dypte=np.int64)
    out_vector[:len(idxs)] = idxs
    out_vector[len(idxs):] = self.cbow_vocab.mask_idx

    return out_vector

  @classmethod
  def from_dataframe(cls, df: pd.DataFrame) -> Vectorizer:
    """
      Instantiate the vectorizer from dataset dataframe

      Args:
        df: target dataset

      Returns:
        an instance of the vectorizer
    """
    cbow_vocab = Vocabulary()
    for idx, row in df.iterrows():
      for token in row['context'].split(' '):
        cbow_vocab.add_token(token)
      cbow_vocab.add_token(row['target'])

    return cls(cbow_vocab)

  @classmethod
  def from_serializable(cls, contents: dict) -> Vectorizer:
    cbow_vocab = Vocabulary.from_serializable(contents['cbow_vocab'])
    return cls(cbow_vocab=cbow_vocab)

  def to_serializable(self) -> dict:
    return {'cbow_vocab': self.cbow_vocab.to_serializable()}

