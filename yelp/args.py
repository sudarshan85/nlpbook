#!/usr/bin/env python

from argparse import Namespace

args = Namespace(
    frequency_cutoff=25,
    workdir_name='scratch',
    vectorizer_fname='vectorizer.json',
    batch_size=128,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=100,
    device='cuda:3',
    sample_file='reviews_with_splits_lite.csv',
    checkpointer_prefix='yelp',
    checkpointer_name='classifier'
    )
