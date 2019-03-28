#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

# consts are loaded into one directory above
path = Path('../data/ag_news')
work_dir=path/'work_dir'

consts = Namespace(
    path=path,
    work_dir=work_dir,
    raw_dataset_csv=path/'news.csv',
    proc_dataset_csv=path/'news_with_splits.csv',
    model_dir=work_dir/'models',
    vectorizer_file=work_dir/'vectorizer.json',
    metric_file=work_dir/'metrics.csv',
    embedding_size=100,
    batch_size=64,
    learning_rate=0.0001,
    num_epochs=100,
    device='cuda:3',
    checkpointer_prefix='cbow',
    checkpointer_name='classifier',
    early_stopping_criteria=5,
    save_every=2, # save model every n epochs
    save_total=5, # have total of n models saved
    )
