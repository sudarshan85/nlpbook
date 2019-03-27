#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

# args are loaded into one directory above
path = Path('../data/books')
work_dir=path/'work_dir'

consts = Namespace(
    path=path,
    work_dir=work_dir,
    proc_dataset_csv=path/'frankenstein_with_splits.csv',
    model_dir=work_dir/'models',
    vectorizer_file=work_dir/'vectorizer.json',
    metric_file=work_dir/'metrics.csv',
    embedding_size=100,
    batch_size=512,
    learning_rate=0.0001,
    num_epochs=100,
    device='cuda:3',
    checkpointer_prefix='cbow',
    checkpointer_name='cbow_classifier',
    early_stopping_criteria=5,
    save_every=1, # save model every n epochs
    save_total=3, # have total of n models saved
    )
