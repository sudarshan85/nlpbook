#!/usr/bin/env python

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import RunningAverage
from ignite.handlers import EarlyStopping, ModelCheckpoint

class YelpTrainer(object):
  def __init__(self, model, optimizer, loss_fn, train_dl, valid_dl, args, pbar, metrics={}):
    # retrieve required params from args
    self.save_dir = args.save_dir
    self.patience = args.early_stopping_criteria
    self.n_epochs = args.num_epochs
    self.device = args.device
    self.prefix = args.checkpointer_prefix
    self.model_name = args.checkpointer_name

    self.trainer = create_supervised_trainer(model, optimizer, loss_fn, device=self.device)
    self.train_eval = create_supervised_evaluator(model,  metrics=metrics, device=self.device)
    self.val_eval = create_supervised_evaluator(model,  metrics=metrics, device=self.device)

    self.train_dl = train_dl
    self.val_dl = valid_dl
    self.pbar = pbar

    RunningAverage(output_transform=lambda x: x).attach(self.trainer, 'loss')
    self.pbar.attach(self.trainer, ['loss'])

    early_stopping = EarlyStopping(patience=self.patience, score_function=self.score_fn,
        trainer=self.trainer)
    checkpointer = ModelCheckpoint(self.save_dir, self.prefix, save_interval=1, n_saved=2,
        save_as_state_dict=True)

    self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.log_epoch)
    self.trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {self.model_name: model})
    self.val_eval.add_event_handler(Events.COMPLETED, early_stopping)

  def log_epoch(self, engine):
    self.train_eval.run(self.train_dl)
    self.val_eval.run(self.val_dl)
    train_metric = self.train_eval.state.metrics
    val_metric = self.val_eval.state.metrics
    self.pbar.log_message(f"Epoch: {engine.state.epoch}")
    self.pbar.log_message(f"Training - Loss: {train_metric['loss']:0.3f}, Accuracy: {train_metric['accuracy']:0.3f}")
    self.pbar.log_message(f"Validation - Loss: {val_metric['loss']:0.3f}, Accuracy: {val_metric['accuracy']:0.3f}")

  def run(self):
    self.trainer.run(self.train_dl, self.n_epochs)

  @staticmethod
  def score_fn(engine):
    val_loss = engine.state.metrics['loss']
    return -val_loss

