#!/usr/bin/env python

from argparse import Namespace

args = Namespace(
    raw_train_csv='raw_train.csv',
    raw_test_csv='raw_test.csv',
    train_proportion=0.7,
    full_file='reviews_with_splits_full.csv',
    frequency_cutoff=25, # minimum frequency for words to be counted
    lite_dir='models/lite',
    full_dir='models/full',
    vectorizer_fname='vectorizer.json',
    batch_size=1024,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=100,
    device='cuda:2',
    lite_file='reviews_with_splits_lite.csv',
    checkpointer_prefix='yelp',
    checkpointer_name='classifier',
    save_every=2, # save model every n epochs
    save_total=5 # have total of n models saved
    )
