#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

# consts are loaded into one directory above
path = Path('../data/ag_news')
pretrained_path = Path('../pretrained/glove6B')
work_dir=path/'work_dir'

consts = Namespace(
    path=path,
    work_dir=work_dir,
    proc_dataset_csv=path/'news_with_splits.csv',
    model_dir=work_dir/'models',
    vectorizer_file=work_dir/'vectorizer.json',
    metric_file=work_dir/'metrics.csv',
    cw_file=work_dir/'class_weights.pth',
    use_glove=False,
    glove_path=pretrained_path/'glove.6B.100d.txt',
    embedding_size=100,
    hidden_dim=100,
    n_channels=100,
    dropout_p=0.1,
    batch_size=256,
    learning_rate=0.001,
    num_epochs=100,
    device='cuda:3',
    checkpointer_prefix='cbow',
    checkpointer_name='classifier',
    early_stopping_criteria=5,
    save_every=2, # save model every n epochs
    save_total=5, # have total of n models saved
    )
