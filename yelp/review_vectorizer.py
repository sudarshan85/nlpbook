#!/usr/bin/env python

import pandas as pd
import numpy as np
import string

from collections import Counter

from vocab import Vocabulary

class ReviewVectorizer(object):
  """
    The Vectorizer which coordinates the vocabularies and puts them to use
  """
  def __init__(self, review_vocab: Vocabulary, rating_vocab: Vocabulary) -> None:
    """
      Args:
        review_vocab: maps words to integers
        rating_vocab: maps class labels to integers
    """

    self._review_vocab = review_vocab
    self._rating_vocab = rating_vocab


  def vectorize(self, review: str) -> np.ndarray:
    """
      Create a collapsed one-hot vector for the review

      Args:
        review: the review

      Returns:
        one_hot: the collapsed one-hot encoding
    """

    one_hot = np.zeros(len(self._review_vocab), dtype=np.float32)

    for token in review.split(' '):
      if token not in string.punctuation:
        one_hot[self._review_vocab.lookup_token(token)] = 1

    return one_hot


  @classmethod
  def from_dataframe(cls, review_df: pd.DataFrame, cutoff: int = 25) -> ReviewVectorizer:
    """
      Instantiates a ReviewVectorizer object from the dataset dataframe

      Args:
        review_df: the review dataset
        cutoff   : the parameter for frequency-based filtering

      Returns:
        an instance of the ReviewVectorizer
    """

    review_vocab = Vocabulary(add_unk=True)
    rating_vocab = Vocabulary(add_unk=False)

    # Add ratings
    for rating in sorted(set(review_df['rating'])):
      rating_vocab.add_token(rating)

    # Add top words if count > provided count
    word_counts: Counter = Counter()
    for review in review_df['review']:
      for word in review.split(' '):
        if word not in string.punctuation:
          word_counts[word] += 1

    for word, count in word_counts.items():
      if count > cutoff:
        review_vocab.add_token(word)

    return cls(review_vocab, rating_vocab)

  @classmethod
  def from_serializable(cls, contents: dict) -> ReviewVectorizer:
    """
      Instantiates a ReviewVectorizer from a serializable dictionary
    """
    review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
    rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])

    return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)

  def to_serializeable(self) -> dict:
    """
      Create the serializable dictionary for caching
    """
    return {
        'review_vocab': self._review_vocab.to_serializeable(),
        'rating_vocab': self._rating_vocab.to_serializeable()
        }


