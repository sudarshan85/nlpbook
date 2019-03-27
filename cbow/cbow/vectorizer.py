#!/usr/bin/env python

import numpy as np
import json

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



