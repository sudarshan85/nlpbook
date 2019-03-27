#!/usr/bin/env python

from typing import List
from bidict import bidict

class Vocabulary(object):
  def __init__(self, idx_token_bidict: bidict=None, mask_token: str='<MASK>', add_unk: bool=True,
      unk_token: str='<UNK>') -> None:
    if not idx_token_bidict:
      idx_token_bidict = bidict()

    self.idx_token_bidict = idx_token_bidict
    self.size = len(self.idx_token_bidict)
    self.add_unk = add_unk
    self.mask_token = mask_token
    self.unk_token = unk_token

    self.unk_idx = -1
    if add_unk:
      self.unk_idx = self.add_token(self.unk_token)

    self.mask_idx = self.add_token(self.mask_token)

  def add_token(self, token: str) -> int:
    if token in self.idx_token_bidict.values():
      idx = self.idx_token_bidict.inverse[token]
    else:
      idx = len(self.idx_token_bidict)
      self.idx_token_bidict.put(idx, token)
      self.size += 1

    return idx

  def add_many(self, tokens: List[str]) -> List[int]:
    return [self.add_token(token) for token in tokens]

  def lookup_token(self, token: str) -> int:
    if self.unk_idx >= 0:
      if token in self.idx_token_bidict.values():
        return self.idx_token_bidict.inverse[token]
      else:
        return self.unk_idx
    else:
      return self.idx_token_bidict.inverse[token]

  def lookup_idx(self, idx: int) -> str:
    if idx not in self.idx_token_bidict:
      raise KeyError(f"The index {idx} is no in the vocabulary")
    return self.idx_token_bidict[idx]

  def to_serializable(self):
    return {
        'idx_token_bidict': self.idx_token_bidict,
        'add_unk': self.add_unk,
        'unk_token': self.unk_token,
        'mask_token': self.mask_token
        }

  @classmethod
  def from_serializable(self):
    return cls(**contents)

  def __repr__(self):
    return f'<Vocabulary(size={self.size})'

  def __len__(self):
    return self.size

