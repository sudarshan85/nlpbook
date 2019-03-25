#!/usr/bin/env python

import csv
from pathlib import Path

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import RunningAverage
from ignite.handlers import EarlyStopping, ModelCheckpoint

class YelpTrainer(object):
  def __init__(self, model_bundle, data_bundle, args, pbar, metrics={}):
    # retrieve required params from args
    self.save_dir = args.save_dir
    self.patience = args.early_stopping_criteria
    self.n_epochs = args.num_epochs
    self.device = args.device
    self.prefix = args.checkpointer_prefix
    self.model_name = args.checkpointer_name

    # get model and data details
    self.module = model_bundle['module']
    self.optimizer = model_bundle['optimizer']
    self.scheduler = model_bundle['scheduler']
    self.loss_fn = model_bundle['loss_fn']

    self.train_dl = data_bundle['train_dl']
    self.val_dl = data_bundle['val_dl']

    # create trainers and evaluators
    self.trainer = create_supervised_trainer(self.module, self.optimizer, self.loss_fn,
        device=self.device)
    self.train_eval = create_supervised_evaluator(self.module,  metrics=metrics, device=self.device)
    self.valid_eval = create_supervised_evaluator(self.module,  metrics=metrics, device=self.device)

    self.pbar = pbar
    self.metrics_file = Path(self.save_dir/'metrics.csv')

    # set loss to be shown in progress bar
    RunningAverage(output_transform=lambda x: x).attach(self.trainer, 'loss')
    self.pbar.attach(self.trainer, ['loss'])

    # setup early stopping and checkpointer
    early_stopping = EarlyStopping(patience=self.patience, score_function=self.score_fn,
        trainer=self.trainer)
    checkpointer = ModelCheckpoint(self.save_dir, self.prefix, require_empty=False, save_interval=2,
        n_saved=5, save_as_state_dict=True)

    # add all the event handlers
    self.trainer.add_event_handler(Events.STARTED, self.open_csv)
    self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.log_epoch)
    self.trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {self.model_name:
      self.module})
    self.trainer.add_event_handler(Events.COMPLETED, self.close_csv)
    self.valid_eval.add_event_handler(Events.COMPLETED, early_stopping)
    self.valid_eval.add_event_handler(Events.COMPLETED, self.scheduler_step)
    # self.trainer.add_event_handler(Events.ITERATION_COMPLETED, self.log_training_loss)

  def scheduler_step(self, engine):
    self.scheduler.step(engine.state.metrics['loss'])

  def open_csv(self, engine):
    self.fp = open(self.metrics_file, 'w')
    self.writer = csv.writer(self.fp)
    row = ['epoch', 'training_loss', 'training_acc', 'validation_loss', 'validation_acc']
    self.writer.writerow(row)

  def log_training_loss(self, engine):
    iteration = (engine.state.iteration-1) % len(self.train_dl) + 1
    if iteration % 100 == 0:
      self.pbar.log_message(f"ITERATION - loss: {engine.state.output:0.4f}")

  def log_epoch(self, engine):
    self.train_eval.run(self.train_dl)
    self.valid_eval.run(self.val_dl)
    epoch = engine.state.epoch

    train_metric = self.train_eval.state.metrics
    valid_metric = self.valid_eval.state.metrics

    train_loss = f"{self.train_eval.state.metrics['loss']:0.3f}"
    train_acc = f"{self.train_eval.state.metrics['accuracy']:0.3f}"
    valid_loss = f"{self.valid_eval.state.metrics['loss']:0.3f}"
    valid_acc = f"{self.valid_eval.state.metrics['accuracy']:0.3f}"

    self.pbar.log_message(f"Epoch: {epoch}")
    self.pbar.log_message(f"Training - Loss: {train_loss}, Accuracy: {train_acc}")
    self.pbar.log_message(f"Validation - Loss: {valid_loss}, Accuracy: {valid_acc}")

    row = [epoch, f"{train_loss}", f"{train_acc}", f"{valid_loss}", f"{valid_acc}"]
    self.writer.writerow(row)

  def close_csv(self, engine):
    self.fp.close()

  def run(self):
    self.trainer.run(self.train_dl, self.n_epochs)

  @staticmethod
  def score_fn(engine):
    valid_loss = engine.state.metrics['loss']
    return -valid_loss

