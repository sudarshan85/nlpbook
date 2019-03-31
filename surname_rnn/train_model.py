#!/usr/bin/env

import logging
import pandas as pd
import torch
import sys

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar

from consts import consts

from surname.dataset import SurnameDataset
from surname.containers import DataContainer, ModelContainer
from surname.model import SurnameClassifier
from surname.trainer import IgniteTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)

if __name__=='__main__':
  logger.info("Loading data...")
  df = pd.read_csv(consts.proc_dataset_csv)
  dc = DataContainer(df, SurnameDataset, consts.vectorizer_json, consts.bs, with_test=True,
      is_load=True)

  try:
      class_weights = torch.load(consts.class_weights_pth)
  except FileNotFoundError:
    nationality_vocab = dc.nationality_vocab
    class_counts = df['nationality'].value_counts().to_dict()
    sorted_counts = sorted(class_counts.items(), key=lambda x:
    nationality_vocab.lookup_token(x[0]))
    freq = [count for _, count in sorted_counts]
    class_weights = 1.0/torch.tensor(freq, dtype=torch.float32)
    torch.save(class_weights, consts.class_weights_pth)

  logger.info("Creating model")
  classifier = SurnameClassifier(consts.char_embedding_sz, dc.vocab_size, dc.n_classes,
      consts.rnn_hidden_sz, padding_idx=dc.surname_vocab.mask_idx)
  optimizer = optim.Adam(classifier.parameters(), lr=consts.lr)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 1)
  class_weights = class_weights.to(consts.device)
  loss_fn = nn.CrossEntropyLoss(class_weights)

  mc = ModelContainer(classifier, optimizer, loss_fn, scheduler)
  pbar = ProgressBar(persist=True)
  metrics = {'accuracy': Accuracy(), 'loss': Loss(loss_fn)}

  logger.info("Running model for {} epochs on device {} with batch size {}"
      .format(consts.n_epochs, consts.device, consts.bs))
  ig = IgniteTrainer(mc, dc, consts, pbar, metrics)
  ig.run()
