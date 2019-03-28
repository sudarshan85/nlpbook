#!/usr/bin/env

import logging
import pandas as pd
import torch

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar

from consts import consts

from cbow.dataset import CBOWDataset, DataContainer
from cbow.model import CBOWClassifier, ModelContainer
from cbow.trainer import IgniteTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)

if __name__=='__main__':
  logger.info("Loading data...")
  df = pd.read_csv(consts.proc_dataset_csv)
  dc = DataContainer(df, consts.vectorizer_file, consts.batch_size, with_test=True, is_load=True)

  logger.info("Instantiating model...")
  classifier = CBOWClassifier(dc.vocabulary_size, consts.embedding_size)
  loss_func = nn.CrossEntropyLoss()
  optimizer = optim.Adam(classifier.parameters(), lr=consts.learning_rate)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1,
      patience=1)

  mc = ModelContainer(classifier, optimizer, loss_func, scheduler)

  pbar = ProgressBar(persist=True)
  metrics = {'accuracy': Accuracy(), 'loss': Loss(loss_func)}

  consts.num_epochs=2
  logger.info("Running model for {} epochs on device {} with batch size {}"
      .format(consts.num_epochs, consts.device, consts.batch_size))
  ig = IgniteTrainer(mc, dc, consts, pbar, metrics)
  ig.run()
