#!/usr/bin/env python

from typing import List
from bidict import bidict

class Vocabulary(object):
  def __init__(self, idx_token_bidict: bidict=None) -> None:
    if not idx_token_bidict:
      idx_token_bidict = bidict()

    self.idx_token_bidict = idx_token_bidict
    self.size = len(self.idx_token_bidict)

  def to_serializable(self):
    return {'idx_token_map': dict(self.idx_token_bidict)}

  @classmethod
  def from_serializable(cls, contents):
    idx_token_map = {int(k): v for k,v in contents['idx_token_map'].items()}
    idx_token_bidict = bidict(idx_token_map)
    return cls(idx_token_bidict)

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
    return self.idx_token_bidict.inverse[token]

  def lookup_idx(self, idx: int) -> str:
    if idx not in self.idx_token_bidict:
      raise KeyError(f"The index {idx} is no in the vocabulary")
    return self.idx_token_bidict[idx]

  def __repr__(self):
    return f'<Vocabulary(size={self.size})'

  def __len__(self):
    return self.size

