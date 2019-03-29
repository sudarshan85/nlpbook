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

from ag.data import DataContainer, NewsDataset
from ag.model import ModelContainer, NewsClassifier
from ag.pretrained_emb import PretrainedEmbeddings
from ag.trainer import IgniteTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(sh)

def vanilla(consts, vocab_size, n_classes):
  classifier = NewsClassifier(consts.embedding_size, vocab_size, consts.n_channels,
      consts.hidden_dim, n_classes, consts.dropout_p)
  optimizer = optim.Adam(classifier.parameters(), lr=consts.learning_rate)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 1)

  return classifier, optimizer, scheduler

def pretrained(consts, vocab_size, n_classes, pretrained):
  classifier = NewsClassifier(consts.embedding_size, vocab_size, consts.n_channels,
      consts.hidden_dim, n_classes, consts.dropout_p, pretrained=pretrained)

  return classifier, optimizer, scheduler

def pretrained_frozen(consts, vocab_size, n_classes, pretrained):
  classifier = NewsClassifier(consts.embedding_size, vocab_size, consts.n_channels,
      consts.hidden_dim, n_classes, consts.dropout_p, pretrained=pretrained, freeze_pretrained=True)
  optimizer = optim.Adam(classifier.parameters(), lr=consts.learning_rate)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 1)

  return classifier, optimizer, scheduler

if __name__=='__main__':
  pretrained = False
  freeze = False

  if len(sys.argv) == 1:
    pass
  elif sys.argv[1] == 'pretrained':
    pretrained = True
  elif sys.argv[1] == 'freeze':
    pretrained = True
    freeze = True

  logger.info("Loading data...")
  df = pd.read_csv(consts.proc_dataset_csv)
  dc = DataContainer(df, NewsDataset, consts.vectorizer_file, consts.batch_size, with_test=True,
      is_load=True)

  try:
    class_weights = torch.load(consts.cw_file)
  except FileNotFoundError:
    cat_vocab = dc.cat_vocab
    class_counts = df['category'].value_counts().to_dict()
    sorted_counts = sorted(class_counts.items(), key=lambda x: cat_vocab.lookup_token(x[0]))
    freq = [count for _, count in sorted_counts]
    class_weights = 1.0/torch.tensor(freq, dtype=torch.float32)
    torch.save(class_weights, consts.cw_file)

  if not pretrained:
    logger.info("Creating model without pretrained embeddings")
    consts.device = 'cuda:0'
    consts.metrics_file = consts.work_dir/'metrics_vanilla.csv'
    consts.checkpointer_name = 'vanilla'
    class_weights = class_weights.to(consts.device)
    # classifier, optimizer, scheduler = vanilla_model(consts, dc.vocab_size, dc.n_classes)
    classifier = NewsClassifier(consts.embedding_size, dc.vocab_size, consts.n_channels,
        consts.hidden_dim, dc.n_classes, consts.dropout_p)
  elif pretrained and not freeze:
    logger.info("Creating model with differentiable pretrained embeddings")
    pe = PretrainedEmbeddings.from_file(consts.glove_path)
    pe.make_custom_embeddings(list(dc.title_vocab.idx_token_bidict.values()))
    consts.device = 'cuda:1'
    consts.metrics_file = consts.work_dir/'metrics_pretrained.csv'
    consts.checkpointer_name = 'pretrained'
    class_weights = class_weights.to(consts.device)
    # classifier, optimizer, scheduler = pretrained_model(consts, dc.vocab_size, dc.n_classes,
        # pe.custom_embeddings)
    classifier = NewsClassifier(consts.embedding_size, dc.vocab_size, consts.n_channels,
        consts.hidden_dim, dc.n_classes, consts.dropout_p, pretrained=pe.custom_embeddings)
  else:
    logger.info("Creating model with frozen pretrained embeddings")
    pe = PretrainedEmbeddings.from_file(consts.glove_path)
    pe.make_custom_embeddings(list(dc.title_vocab.idx_token_bidict.values()))
    consts.device = 'cuda:2'
    consts.metrics_file = consts.work_dir/'metrics_frozen.csv'
    consts.checkpointer_name = 'frozen'
    class_weights = class_weights.to(consts.device)
    # classifier, optimizer, scheduler = pretrained_model(consts, dc.vocab_size, dc.n_classes,
        # pe.custom_embeddings)
    classifier = NewsClassifier(consts.embedding_size, dc.vocab_size, consts.n_channels,
        consts.hidden_dim, dc.n_classes, consts.dropout_p, pretrained=pe.custom_embeddings,
        freeze_pretrained=True)

  optimizer = optim.Adam(classifier.parameters(), lr=consts.learning_rate)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 1)
  loss_func = nn.CrossEntropyLoss(class_weights)
  print(consts)
  print(class_weights)

  # logger.info("Instantiating model...")
  # classifier = CBOWClassifier(dc.vocabulary_size, consts.embedding_size)
  # loss_func = nn.CrossEntropyLoss()
  # optimizer = optim.Adam(classifier.parameters(), lr=consts.learning_rate)
  # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1,
      # patience=1)

  # mc = ModelContainer(classifier, optimizer, loss_func, scheduler)

  # pbar = ProgressBar(persist=True)
  # metrics = {'accuracy': Accuracy(), 'loss': Loss(loss_func)}

  # logger.info("Running model for {} epochs on device {} with batch size {}"
      # .format(consts.num_epochs, consts.device, consts.batch_size))
  # ig = IgniteTrainer(mc, dc, consts, pbar, metrics)
  # ig.run()
