#!/usr/bin/env python

import pandas as pd
import numpy as np

from typing import Tuple

from .vocabulary import Vocabulary, SequenceVocabulary

class Vectorizer(object):
  def __init__(self, char_vocab: Vocabulary, nationality_vocab: Vocabulary):
    self.char_vocab = char_vocab
    self.nationality_vocab = nationality_vocab

  def vectorize(self, surname: str, vector_len: int) -> Tuple[np.ndarray, int]:
    """
      Args:
        surname: input surname
        vector_len: length of the longest surname

      Returns:
        vectorizerd surname and the sequence length
    """
    bos = [self.char_vocab.bos_idx]
    eos = [self.char_vocab.eos_idx]
    surname_idxs = [self.char_vocab.lookup_token(char) for char in surname]
    idxs = bos + surname_idxs + eos
    seq_len = len(idxs)

    out_vector = np.zeros(vector_len, dtype=np.int64)
    out_vector[:seq_len] = idxs
    out_vector[seq_len:] = self.char_vocab.mask_idx

    return out_vector, seq_len

  @classmethod
  def from_dataframe(cls, df: pd.DataFrame):
    """
      Instantiate the vectorizer from dataset dataframe

      Args:
        df: target dataset

      Returns:
        an instance of the vectorizer
    """
    char_vocab = SequenceVocabulary()
    nationality_vocab = Vocabulary()
    nationality_vocab.add_many(list(df['nationality'].unique()))

    for idx, row in df.iterrows():
      for char in df['surname']:
        char_vocab.add_token(char)

    return cls(char_vocab, nationality_vocab)

  @classmethod
  def from_serializable(cls, contents: dict):
    char_vocab = SequenceVocabulary.from_serializable(contents['char_vocab'])
    nationality_vocab = Vocabulary.from_serializable(contents['nationality_vocab'])
    return cls(char_vocab, nationality_vocab)

  def to_serializable(self) -> dict:
    return {
        'char_vocab': self.char_vocab.to_serializable(),
        'nationality_vocab': self.nationality_vocab.to_serializable(),
        }

