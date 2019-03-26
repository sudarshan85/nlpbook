#!/usr/bin/env

import numpy as np
from pathlib import Path
from typing import List
from bidict import bidict
from annoy import AnnoyIndex

class PretrainedEmbeddings(object):
  """
    A wrapper around pre-trained word vectors
  """
  def __init__(self, idx_word_bidict: bidict, word_vectors: List[np.ndarray]):
    """
      idx_word_map: a bidict mapping between indices and words
      word_vectors: list of numpy arrays
    """
    self.idx_word_bidict = idx_word_bidict
    self.word_vectors = word_vectors
    self.vocab_size = len(word_vectors)
    self.embedding_size = word_vectors[0].shape[0]

    self.idx = AnnoyIndex(self.embedding_size, metric='euclidean')
    print("Building Index...")
    for i, _ in self.idx_word_bidict.items():
      self.idx.add_item(i, self.word_vectors[i])
    self.idx.build(50)
    print("Finished")

  @classmethod
  def from_file(cls, embedding_file: Path):
    """
      Instantial from pre-trained vector file

      Vector file should be of the format:
        word0 x0_0 x0_1 ... x0_N
        word1 x1_0 x1_1 ... x1_N

      Args:
        embedding_file: location of the file
      Returns:
        Instance of PretrainedEmbeddings
    """
    idx_word_bidict = bidict()
    word_vectors = []

    print("Loading file...")
    with open(embedding_file, 'r') as fp:
      for line in fp.readlines():
        line = line.split(' ')
        word = line[0]
        vec = np.array([float(x) for x in line[1:]])

        idx_word_bidict.put(len(idx_word_bidict), word)
        word_vectors.append(vec)

    return cls(idx_word_bidict, word_vectors)

  def get_embedding(self, word: str) -> np.ndarray:
    return self.word_vectors[self.idx_word_bidict.inverse[word]]

  def get_neighbors(self, vector: np.ndarray, n: int=1) -> List[str]:
    """
      Given a vector, return its n nearest neighbors

      Args:
        vector: should match the size of the vectors in the Annoy idx
    """
    nn_idxs = self.idx.get_nns_by_vector(vector, n)
    return [self.idx_word_bidict[neighbor_idx] for neighbor_idx in nn_idxs]

  def __len__(self):
    return self.embedding_size

  def get_analogy(self, word1, word2, word3) -> str:
    """
      Computes solutions to analogies using word embeddings

      Analogies are word1 is to word2 as word3 is to ____
    """
    # get embedding of 3 words
    vec1 = self.get_embedding(word1)
    vec2 = self.get_embedding(word2)
    vec3 = self.get_embedding(word3)

    # compute 4th word embedding
    spatial_relationship = vec2 - vec1
    vec4 = vec3 + spatial_relationship

    closest_words = self.get_neighbors(vec4, n=4)
    existing_words = set([word1, word2, word3])
    closest_words = [word for word in closest_words if word not in existing_words]

    if len(closest_words) == 0:
      return 'Could not find nearest neighbors for computed vector!'

    words = []
    for word4 in closest_words:
      words.append(f'{word1} : {word2} :: {word3} : {word4}')

    return '\n'.join(words)

  def __repr__(self):
    return f"Pretrained Embeddings of {self.embedding_size} dimensions with {self.vocab_size} words"

if __name__=='__main__':
  GLOVE6B_300 = Path('../pretrained/glove.6B.300d.txt')
  embeddings = PretrainedEmbeddings.from_file(GLOVE6B_300)
  print(embeddings)
  print("--------------------------")
  print(embeddings.get_analogy(*['man', 'he', 'woman']))
  print("--------------------------")
  print(embeddings.get_analogy(*['man', 'uncle', 'woman']))
  print("--------------------------")
  print(embeddings.get_analogy(*['talk', 'communicate', 'read']))
  print("--------------------------")
  print(embeddings.get_analogy(*['man', 'king', 'woman']))
  print("--------------------------")
  print(embeddings.get_analogy(*['king', 'queen', 'husband']))

