#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

# consts are loaded into one directory above
path = Path('../data/surnames')
work_dir=path/'rnn_workdir'

consts = Namespace(
    path=path,
    work_dir=work_dir,
    proc_dataset_csv=path/'surnames_with_splits.csv',
    model_dir=work_dir/'models',
    vectorizer_file=work_dir/'vectorizer.json',
    metrics_file=work_dir/'metrics.csv',
    cw_file=work_dir/'class_weights.pth',
    embedding_size=100,
    hidden_dim=103,
    batch_size=128,
    learning_rate=0.001,
    num_epochs=97,
    device='cuda:3',
    checkpointer_prefix='surname_elman',
    checkpointer_name='classifier',
    early_stopping_criteria=11,
    save_every=2, # save model every n epochs
    save_total=5, # have total of n models saved
    )
