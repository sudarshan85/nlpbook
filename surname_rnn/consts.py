#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

# consts are loaded into one directory above
path = Path('../data/surnames')
workdir=path/'rnn_workdir'

consts = Namespace(
    path=path,
    workdir=workdir,
    proc_dataset_csv=path/'surnames_with_splits.csv',
    model_dir=workdir/'models',
    vectorizer_json=workdir/'elman_vectorizer.json',
    metrics_file=workdir/'elman_metrics.csv',
    class_weights_pth=workdir/'class_weights.pth',
    char_embedding_sz=100,
    rnn_hidden_sz=64,
    bs=64,
    lr=1e-3,
    n_epochs=97,
    device='cuda:3',
    checkpointer_prefix='elman',
    checkpointer_name='classifier',
    es_patience=11,
    save_every=2, # save model every n epochs
    save_total=5, # have total of n models saved
    )
