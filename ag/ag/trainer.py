#!/usr/bin/env python

import csv
import datetime
from argparse import Namespace
from pathlib import Path

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import RunningAverage
from ignite.handlers import EarlyStopping, ModelCheckpoint, Timer
from ignite.contrib.handlers import ProgressBar

from .model import ModelContainer
from .data import DataContainer

class IgniteTrainer(object):
  def __init__(self, mc: ModelContainer, dc: DataContainer, consts: Namespace, pbar:
    ProgressBar, metrics: dict={}) -> None:

    # retreive required constants from consts
    self.model_dir = consts.model_dir
    self.metrics_file = consts.metrics_file
    self.patience = consts.early_stopping_criteria
    self.n_epochs = consts.num_epochs
    self.device = consts.device
    self.prefix = consts.checkpointer_prefix
    self.model_name = consts.checkpointer_name
    self.save_interval = consts.save_every
    self.n_saved = consts.save_total

    # get model and data details
    self.model = mc.model
    self.optimizer = mc.optimizer
    self.scheduler = mc.scheduler
    self.loss_fn = mc.loss_fn
    self.train_dl = dc.train_dl
    self.val_dl = dc.val_dl

    # create trainers and evaluators
    self.trainer = create_supervised_trainer(self.model, self.optimizer, self.loss_fn,
        device=self.device)
    self.train_eval = create_supervised_evaluator(self.model,  metrics=metrics, device=self.device)
    self.val_eval = create_supervised_evaluator(self.model,  metrics=metrics, device=self.device)

    # set loss to be shown in progress bar
    self.pbar = pbar
    RunningAverage(output_transform=lambda x: x).attach(self.trainer, 'loss')
    self.pbar.attach(self.trainer, ['loss'])

    # setup timers
    self.epoch_timer = Timer(average=True)
    self.epoch_timer.attach(self.trainer, start=Events.EPOCH_COMPLETED, resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
    self.training_timer = Timer()
    self.training_timer.attach(self.trainer)

    # setup early stopping and checkpointer
    early_stopping = EarlyStopping(patience=self.patience, score_function=self.score_fn,
        trainer=self.trainer)
    checkpointer = ModelCheckpoint(self.model_dir, self.prefix, require_empty=False,
        save_interval=self.save_interval, n_saved=self.n_saved, save_as_state_dict=True)

    # add all the event handlers
    self.trainer.add_event_handler(Events.STARTED, self._open_csv)
    self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self._log_epoch)
    self.trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {self.model_name:
      self.model})
    self.trainer.add_event_handler(Events.COMPLETED, self._close_csv)
    self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self._log_training_loss)

    # self.val_eval.add_event_handler(Events.COMPLETED, early_stopping)
    self.val_eval.add_event_handler(Events.COMPLETED, self._scheduler_step)

  def _open_csv(self, engine):
    self.fp = open(self.metrics_file, 'w')
    self.writer = csv.writer(self.fp)
    row = ['epoch', 'training_loss', 'training_acc', 'validation_loss', 'validation_acc']
    self.writer.writerow(row)

  def _scheduler_step(self, engine):
    self.scheduler.step(engine.state.metrics['loss'])

  def _log_training_loss(self, engine):
    iteration = (engine.state.iteration-1) % len(self.train_dl) + 1
    if iteration % 100 == 0:
      self.pbar.log_message(f"ITERATION - loss: {engine.state.output:0.4f}")

  def _log_epoch(self, engine):
    self.epoch_timer.reset()
    self.train_eval.run(self.train_dl)
    self.val_eval.run(self.val_dl)
    epoch = engine.state.epoch

    train_metric = self.train_eval.state.metrics
    valid_metric = self.val_eval.state.metrics

    train_loss = f"{self.train_eval.state.metrics['loss']:0.3f}"
    train_acc = f"{self.train_eval.state.metrics['accuracy']:0.3f}"
    valid_loss = f"{self.val_eval.state.metrics['loss']:0.3f}"
    valid_acc = f"{self.val_eval.state.metrics['accuracy']:0.3f}"

    self.pbar.log_message(f"Epoch: {epoch}")
    self.pbar.log_message(f"Training - Loss: {train_loss}, Accuracy: {train_acc}")
    self.pbar.log_message(f"Validation - Loss: {valid_loss}, Accuracy: {valid_acc}")
    self.pbar.log_message(f"Time per batch {self.epoch_timer.value():0.3f}[s]")

    row = [epoch, f"{train_loss}", f"{train_acc}", f"{valid_loss}", f"{valid_acc}"]
    self.writer.writerow(row)

  def _close_csv(self, engine):
    train_time = str(datetime.timedelta(seconds=self.training_timer.value()))
    self.pbar.log_message(f"Training Done. Total training time: {train_time}")
    self.fp.write(f"{train_time}\n")
    self.fp.close()

  def run(self):
    self.trainer.run(self.train_dl, self.n_epochs)

  @staticmethod
  def score_fn(engine):
    valid_loss = engine.state.metrics['loss']
    return -valid_loss

