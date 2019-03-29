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
    classifier = NewsClassifier(consts.embedding_size, dc.vocab_size, consts.n_channels,
        consts.hidden_dim, dc.n_classes, consts.dropout_p, pretrained=pe.custom_embeddings,
        freeze_pretrained=True)

  optimizer = optim.Adam(classifier.parameters(), lr=consts.learning_rate)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 1)
  loss_func = nn.CrossEntropyLoss(class_weights)

  mc = ModelContainer(classifier, optimizer, loss_func, scheduler)
  pbar = ProgressBar(persist=True)
  metrics = {'accuracy': Accuracy(), 'loss': Loss(loss_func)}

  consts.num_epochs=1
  logger.info("Running model for {} epochs on device {} with batch size {}"
      .format(consts.num_epochs, consts.device, consts.batch_size))
  ig = IgniteTrainer(mc, dc, consts, pbar, metrics)
  ig.run()
