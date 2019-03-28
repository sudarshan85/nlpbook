#!/usr/bin/env python

import pandas as pd
import numpy as np
import string

from collections import Counter

from .vocabulary import Vocabulary

class Vectorizer(object):
  def __init__(self, title_vocab: Vocabulary, category_vocab: Vocabulary):
    self.title_vocab = title_vocab
    self.category_vocab = category_vocab

  # def vectorizer(self, title: str, vector_len: int=-1) -> np.ndarray:
  def vectorizer(self, title: str, vector_len: int) -> np.ndarray:
    """
      Args:
        title: string of words separated by a space
        vector_len: an argument for forcing the length of index vector

      Returns:
        vectorized title
    """
    bos = [self.title_vocab.bos_idx]
    eos = [self.title_vocab.eos_idx]
    idxs = [self.title_vocab.lookup_token(token) for token in title.split(' ')]
    vector = bos + idxs + eos

    # if vector_len < 0:
      # vector_len = len(vector)

    out_vector = np.zeros(vector_len, dtype=np.int64)
    out_vector[:len(vector)] = vector
    out_vector[len(vector):] = self.title_vocab.mask_idx

    return out_vector

  @classmethod
  def from_dataframe(cls, df: pd.DataFrame, cutoff: int=25):
    """
      Instantiate the vectorizer from dataset dataframe

      Args:
        df: target dataset

      Returns:
        an instance of the vectorizer
    """
    title_vocab = Vocabulary()
    category_vocab = Vocabulary()
    category_vocab.add_many(list(df['category'].unique()))

    word_counts: Counter = Counter()
    for title in df['title']:
      for word in title.split(' '):
        if word not in string.punctuation:
          word_counts[word] += 1

    for word, count in word_counts.items():
      if count > cutoff:
        title_vocab.add_token(word)

    return cls(title_vocab, category_vocab)

  @classmethod
  def from_serializable(cls, contents: dict):
    title_vocab = Vocabulary.from_serializable(contents['title_vocab'])
    category_vocab = Vocabulary.from_serializable(contents['category_vocab'])
    return cls(title_vocab, category_vocab)

  def to_serializable(self) -> dict:
    return {
        'title_vocab': self.title_vocab.to_serializable(),
        'category_vocab': self.category_vocab.to_serializable(),
        }

