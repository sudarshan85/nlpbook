#!/usr/bin/env python

import pandas as pd
import numpy as np

from typing import Tuple

from .vocabulary import Vocabulary, SequenceVocabulary

class Vectorizer(object):
  def __init__(self, surname_vocab: Vocabulary, nationality_vocab: Vocabulary):
    self.surname_vocab = surname_vocab
    self.nationality_vocab = nationality_vocab

  @classmethod
  def from_dataframe(cls, df: pd.DataFrame):
    """
      Instantiate the vectorizer from dataset dataframe

      Args:
        df: target dataset

      Returns:
        an instance of the vectorizer
    """
    surname_vocab = SequenceVocabulary()
    nationality_vocab = Vocabulary()
    nationality_vocab.add_many(list(df['nationality'].unique()))

    for idx, row in df.iterrows():
      for char in row['surname']:
        surname_vocab.add_token(char)

    return cls(surname_vocab, nationality_vocab)

  @classmethod
  def from_serializable(cls, contents: dict):
    surname_vocab = SequenceVocabulary.from_serializable(contents['surname_vocab'])
    nationality_vocab = Vocabulary.from_serializable(contents['nationality_vocab'])
    return cls(surname_vocab, nationality_vocab)

  def to_serializable(self) -> dict:
    return {
        'surname_vocab': self.surname_vocab.to_serializable(),
        'nationality_vocab': self.nationality_vocab.to_serializable(),
        }

class GRUVectorizer(Vectorizer):
  def __init(self, surname_vocab: Vocabulary, nationality_vocab: Vocabulary) -> None:
    super(GRUVectorizer, self).__init__(surname_vocab, nationality_vocab)

  def vectorizer(self, surname: str, vector_len: int=-1) -> Tuple[np.ndarray, np.ndarray]:
    """
      Vectorizer a surname into a vector of observantions
    """
    bos = [self.surname_vocab.bos_idx]
    eos = [self.surname_vocab.eos_idx]
    surname_idxs = [self.surname_vocab.lookup_token(char) for char in surname]
    idxs = bos + surname_idxs + eos
    seq_len = len(idxs)


class ClassificationVectorizer(Vectorizer):
  def __init__(self, surname_vocab: Vocabulary, nationality_vocab: Vocabulary) -> None:
    super(ClassificationVectorizer, self).__init__(surname_vocab, nationality_vocab)

  def vectorize(self, surname: str, vector_len: int) -> Tuple[np.ndarray, int]:
    """
      Args:
        surname: input surname
        vector_len: length of the longest surname

      Returns:
        vectorizerd surname and the sequence length
    """
    bos = [self.surname_vocab.bos_idx]
    eos = [self.surname_vocab.eos_idx]
    surname_idxs = [self.surname_vocab.lookup_token(char) for char in surname]
    idxs = bos + surname_idxs + eos
    seq_len = len(idxs)

    out_vector = np.zeros(vector_len, dtype=np.int64)
    out_vector[:seq_len] = idxs
    out_vector[seq_len:] = self.surname_vocab.mask_idx

    return out_vector, seq_len
