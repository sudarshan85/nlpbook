#!/usr/bin/env python

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping

class YelpTrainer(object):
  def __init__(self, model,
      optimizer,
      loss_fn,
      train_dl,
      valid_dl,
      pbar,
      metrics={},
      device='cpu'):
    self.trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    self.train_eval = create_supervised_evaluator(model,  metrics=metrics, device=device)
    self.val_eval = create_supervised_evaluator(model,  metrics=metrics, device=device)

    self.train_dl = train_dl
    self.val_dl = valid_dl
    self.pbar = pbar
    self.pbar.attach(self.trainer)

    # early_stopping = EarlyStopping(patience=3, score_function=self.score_fn, trainer=self.trainer)

    self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.log_epoch)

  def log_epoch(self, engine):
    self.train_eval.run(self.train_dl)
    self.val_eval.run(self.val_dl)
    train_metric = self.train_eval.state.metrics
    val_metric = self.val_eval.state.metrics
    self.pbar.log_message(f"Epoch: {engine.state.epoch}, Avg Accuracy: train {train_metric['accuracy']:0.3f} valid {val_metric['accuracy']:0.3f} Avg Loss: train {train_metric['bce']:0.3f} valid {val_metric['bce']:0.3f}")

  def run(self, epochs):
    self.trainer.run(self.train_dl, epochs)

  # @staticmethod
  # def score_fn(engine):
    # val_loss = engine.state.metrics['bce']
    # return -val_lossj

