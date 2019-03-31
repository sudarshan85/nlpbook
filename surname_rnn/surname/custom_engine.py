#!/usr/bin/env python

"""
  Custom Ignite process function to extract data correctly from the dataset and run it through the
  model. Lightly modified from original supervised trainer and evaluators in __init__.py
"""

import torch

from torch import nn
from torch import optim
from typing import Callable

from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor

def _prepare_batch(batch, device: str=None, non_blocking: bool=False):
  inp,y = batch
  return (
      (convert_tensor(inp[0], device=device, non_blocking=non_blocking),
      convert_tensor(inp[1], device=device, non_blocking=non_blocking)),
      convert_tensor(y, device=device, non_blocking=non_blocking)
      )

def custom_trainer(model: nn.Module, optimizer: optim, loss_fn: Callable, device: str=None,
    non_blocking: bool=False, prepare_batch=_prepare_batch) -> Engine:
  if device:
    model.to(device)

  def _update(engine, batch) -> float:
    model.train()
    optimizer.zero_grad()
    inp,y = prepare_batch(batch, device, non_blocking=non_blocking)
    y_pred = model(*inp)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

  return Engine(_update)

def custom_evaluator(model: nn.Module, metrics: dict={}, device: str=None, non_blocking: bool=False,
    prepare_batch=_prepare_batch) -> Engine:
  if device:
    model.to(device)

  def _inference(engine, batch):
    model.eval()
    with torch.no_grad():
      inp,y = prepare_batch(batch, device, non_blocking=non_blocking)
      y_pred = model(*inp)
      return y_pred, y

  engine = Engine(_inference)

  for name, metric in metrics.items():
    metric.attach(engine, name)

  return engine
