#!/usr/bin/env python

"""
  Module contains the vocabulary class from the NLP with PyTorch book with certain modifications.
  Code can be found: http://bit.ly/2XYExLh
"""

import pdb
from typing import Dict, List

class Vocabulary(object):
  """
    Class to process text and extract vocabulary for mapping
  """
  def __init__(self,
      token2idx: Dict[str, int] = None,
      add_unk: bool = True,
      unk_token: str='xxunk') -> None:
    """
      Args:
        token2idx: pre-extisting map of tokens to indices
        add_unk  : a flag that indicates whether to add the unknown token
        unk_token: the unknown token to add to the vocabulary
    """

    if token2idx is None:
      token2idx = {}

    self._token2idx = token2idx
    self._idx2token = {idx: token for token, idx in self._token2idx.items()}

    self._add_unk = add_unk
    self._unk = unk_token

    self._len = len(self._token2idx)

    self.unk_idx = -1
    if add_unk:
      self.unk_idx = self.add_token(unk_token)


  def add_token(self, token: str) -> int:
    """
      Update mapping dicts based on the token

      Args:
        token: the item to add into the vocabulary

      Returns:
        index: the integer corresponding to the token
    """

    if token in self._token2idx:
      idx = self._token2idx[token]
    else:
      idx = len(self)
      self._token2idx[token] = idx
      self._idx2token[idx] = token
      self._len += 1

    return idx

  def add_many(self, tokens: List[str]) -> List[int]:
    """
      Add a list of tokens into the vocabulary

      Args:
        tokens: a list of string tokens

      Returns:
        indices: a list of indices corresponding to the tokens
    """
    return [self.add_token(token) for token in tokens]


  def lookup_token(self, token: str) -> int:
    """
      Retrieve the index corresponding to the token or the unknown token
      index if token is not present

      Args:
        token: the token to look up

      Returns:
        index: index corresponding to the token
    """
    if self.unk_idx >= 0:
      return self._token2idx.get(token, self.unk_idx)
    else:
      if token not in self._token2idx:
        raise KeyError(f"Token {token} is not in the vocabulary and unknown token not initialized")
      return self._token2idx[token]

  def lookup_idx(self, idx: int) -> str:
    """
      Return the token corresponding to the index
    """
    if idx not in self._idx2token:
      raise KeyError(f"Index '{idx}' is not in the vocabulary")

    return self._idx2token[idx]

  def to_serializeable(self) -> dict:
    """
      Returns a dictionary that can be serialized
    """
    return {'token2idx': self._token2idx, 'add_unk': self._add_unk, 'unk_token': self._unk}

  @classmethod
  def from_serializable(cls, contents: dict):
    """
      Instantiates a vocabulary object from a serialized dictionary
    """
    # pdb.set_trace()
    return cls(**contents)

  def __len__(self) -> int:
    """
      Returns the size of the vocabulary
    """
    return self._len

  def __repr__(self) -> str:
    """
      Returns string with metadata about vocabulary
    """
    return f'<Vocabulary (size={self._len})>'
