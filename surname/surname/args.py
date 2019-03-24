#!/usr/bin/env python

from argparse import Namespace

args = Namespace(
    raw_dataset_csv='surnames.csv',
    train_proportion=0.7,
    proc_dataset_csv='surnames_with_splits.csv',
    model_dir='models',
    vectorizer_fname='vectorizer.json',
    cw_file='class_weights.pt',
    batch_size=64,
    hidden_dim=300,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=100,
    device='cuda:3',
    checkpointer_prefix='surname',
    checkpointer_name='classifier',
    save_every=2, # save model every n epochs
    save_total=5 # have total of n models saved
    )
