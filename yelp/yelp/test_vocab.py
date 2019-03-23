#!/usr/bin/env python

import unittest
import json
from pathlib import Path

from vocab import Vocabulary

tmp_dir = Path('./test_tmp')
tmp_dir.mkdir(exist_ok=True)

class TestVocabulary(unittest.TestCase):
  vocab = Vocabulary(add_unk=True)
  json_path = Path(tmp_dir/'vocab.json')

  def test0_add_token(self):
    self.assertEqual(self.vocab.add_token('hello'), 1)
    self.assertEqual(self.vocab.add_token('world'), 2)

  def test1_add_many(self):
    self.assertEqual(self.vocab.add_many(['this', 'is', 'a', 'test']), [3, 4, 5, 6])

  def test2_len(self):
    self.assertEqual(len(self.vocab), 7)

  def test3_lookup_idx(self):
    self.assertEqual(self.vocab.lookup_idx(1), 'hello')
    self.assertEqual(self.vocab.lookup_idx(2), 'world')
    self.assertEqual(self.vocab.lookup_idx(3), 'this')
    self.assertEqual(self.vocab.lookup_idx(4), 'is')
    self.assertEqual(self.vocab.lookup_idx(5), 'a')
    self.assertEqual(self.vocab.lookup_idx(6), 'test')
    self.assertEqual(self.vocab.lookup_idx(0), 'xxunk')

  def test4_lookup_token(self):
    self.assertEqual(self.vocab.lookup_token('hello'), 1)
    self.assertEqual(self.vocab.lookup_token('world'), 2)
    self.assertEqual(self.vocab.lookup_token('this'), 3)
    self.assertEqual(self.vocab.lookup_token('is'), 4)
    self.assertEqual(self.vocab.lookup_token('a'), 5)
    self.assertEqual(self.vocab.lookup_token('test'), 6)
    self.assertEqual(self.vocab.lookup_token('xxunk'), 0)

  def test5_to_serializeable(self):
    ser = self.vocab.to_serializeable()
    self.assertEqual(list(ser.keys()), ['token2idx', 'add_unk', 'unk_token'])
    with open(self.json_path, 'w') as f:
      json.dump(ser, f)

  def test6_from_serializable(self):
    ser = self.vocab.to_serializeable()
    with open(self.json_path, 'r') as f:
      from_ser = json.load(f)

    self.assertEqual(from_ser, ser)

if __name__=='__main__':
  unittest.main()

