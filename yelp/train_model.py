#!/usr/bin/env python

import pandas as pd
import numpy as np
import torch
import datetime

from pathlib import Path
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar

from yelp.args import args
from yelp.trainer import YelpTrainer
from yelp.model import Classifier
from yelp.dataset import ProjectDataset

path = Path('../data/yelp')

def bce_logits_wrapper(output):
  y_pred, y = output
  y_pred = (torch.sigmoid(y_pred) > 0.5).long()
  return y_pred, y

def get_dataloaders(review_csv):
  df = pd.read_csv(review_csv)

  train_df = df.loc[df['split'] == 'train']
  train_ds = ProjectDataset.load_data_and_vectorizer(train_df, vectorizer_path)
  vectorizer = train_ds.get_vectorizer()
  train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True)

  val_df = df.loc[df['split'] == 'val']
  val_ds = ProjectDataset.load_data_and_vectorizer(val_df, vectorizer_path)
  val_dl = DataLoader(val_ds, args.batch_size, shuffle=True, drop_last=True)

  return train_dl, vectorizer, val_dl

if __name__=='__main__':
  args.save_dir = path/args.full_dir
  review_csv = path/args.full_file

  vectorizer_path = args.save_dir/args.vectorizer_fname
  train_dl, vectorizer, val_dl = get_dataloaders(review_csv)
  print(f"Dataset sizes - Train: {len(train_dl.dataset)}, Valid: {len(val_dl.dataset)}")

  classifier = Classifier(num_features=len((vectorizer).review_vocab))
  optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5,
      patience=1)
  loss_func = nn.BCEWithLogitsLoss()

  pbar = ProgressBar(persist=True)
  metrics = {'accuracy': Accuracy(bce_logits_wrapper), 'loss': Loss(loss_func)}

  yelp_trainer = YelpTrainer(classifier, optimizer, loss_func, train_dl, val_dl, args, pbar,
      metrics)

  t1 = datetime.datetime.now()
  print(f"Started at {t1.strftime('%Y-%m-%d %H:%M:%S')}")
  yelp_trainer.run()
  t2 = datetime.datetime.now()
  dt = t2 - t1
  print(f"Finished at {t2.strftime('%Y-%m-%d %H:%M:%S')}")
  print(f"Took {dt.days} days, {dt.seconds//3600} hours, {(dt.seconds//60)%60} minutes")
