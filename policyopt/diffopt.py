#!/usr/bin/env python3.5

from policyopt.autodiff import *

import numpy as np
import torch

class SGDOptimizer(object):
  def __init__(self, cfg):
    self._cfg = cfg
    self._dim = None

  def reset(self, param_vars, dtype=None):
    if dtype is None:
      dtype = param_vars[0].data.type()
    self._dim = flat_count_vars(param_vars)
    print("DEBUG: sgd: dim:", self._dim)
    self._iter_ct = 0
    self._param = torch.zeros(self._dim).type(dtype)
    self._batch_grad = torch.zeros(self._dim).type(dtype)
    self._grad = torch.zeros(self._dim).type(dtype)
    self._momentum = torch.zeros(self._dim).type(dtype)

  def step(self, param_vars, batch_loss_fn):
    batch_size = self._cfg["batch_size"]
    minibatch_size = self._cfg["minibatch_size"]

    self._grad.zero_()
    sum_batch_ct = 0
    batch_nr = 0
    while sum_batch_ct < minibatch_size:
      batch_ct = min(batch_size, minibatch_size - batch_nr * batch_size)
      batch_loss_var = batch_loss_fn(batch_ct)
      grad_vars = adjoint(batch_loss_var, param_vars)
      assert self._dim == serialize_vars(grad_vars, self._batch_grad)
      self._grad.add_(self._batch_grad)
      sum_batch_ct += batch_ct
      batch_nr += 1
    self._grad /= float(minibatch_size)

    assert self._dim == serialize_vars(param_vars, self._param)

    step_size = self._cfg["step_size"]
    momentum = self._cfg["momentum"]
    nesterov = self._cfg["nesterov"]

    #print("DEBUG: sgd: batch grad:", self._batch_grad.cpu().numpy())
    #print("DEBUG: sgd: grad:      ", self._grad.cpu().numpy())
    #print("DEBUG: sgd: param pre: ", self._param.cpu().numpy())
    if not nesterov:
      self._momentum.copy_(momentum * self._momentum + self._grad)
      self._param.add_(-step_size * self._momentum)
    else:
      raise NotImplementedError
    #print("DEBUG: sgd: param post:", self._param.cpu().numpy())

    assert self._dim == deserialize_vars(param_vars, self._param)

    self._iter_ct += 1

class AdamOptimizer(object):
  def __init__(self, cfg):
    self._cfg = cfg
    self._dim = None

  def reset(self, param_vars, dtype=None):
    if dtype is None:
      dtype = param_vars[0].data.type()
    self._dim = flat_count_vars(param_vars)
    print("DEBUG: adam: dim:", self._dim)
    self._iter_ct = 0
    self._loss = torch.zeros(1).type(dtype)
    self._param = torch.zeros(self._dim).type(dtype)
    self._batch_grad = torch.zeros(self._dim).type(dtype)
    self._grad = torch.zeros(self._dim).type(dtype)
    self._mavg_grad_1 = torch.zeros(self._dim).type(dtype)
    self._mavg_grad_2 = torch.zeros(self._dim).type(dtype)

  def step(self, param_vars, batch_loss_fn):
    batch_size = self._cfg["batch_size"]
    minibatch_size = self._cfg["minibatch_size"]

    self._loss.zero_()
    self._grad.zero_()
    sum_batch_ct = 0
    batch_nr = 0
    while sum_batch_ct < minibatch_size:
      batch_ct = min(batch_size, minibatch_size - batch_nr * batch_size)
      batch_loss_var = batch_loss_fn(batch_ct)
      grad_vars = adjoint(batch_loss_var, param_vars)
      assert self._dim == serialize_vars(grad_vars, self._batch_grad)
      self._loss.add_(batch_loss_var.data)
      self._grad.add_(self._batch_grad)
      sum_batch_ct += batch_ct
      batch_nr += 1
    self._loss /= float(minibatch_size)
    self._grad /= float(minibatch_size)

    if "l2_grad_clip" in self._cfg:
      g = torch.norm(self._grad)
      w = np.minimum(self._cfg["l2_grad_clip"], g) / g
      self._grad.mul_(w)

    assert self._dim == serialize_vars(param_vars, self._param)

    step_size = self._cfg["step_size"]
    decay_rate_1 = self._cfg["decay_rate_1"]
    decay_rate_2 = self._cfg["decay_rate_2"]
    epsilon = self._cfg["epsilon"]

    assert self._iter_ct >= 0
    norm_1 = float(1.0 / (1.0 - np.power(1.0 - decay_rate_1, self._iter_ct + 1)))
    norm_2 = float(1.0 / (1.0 - np.power(1.0 - decay_rate_2, self._iter_ct + 1)))

    #print("DEBUG: adam: batch grad:", self._batch_grad.cpu().numpy())
    #print("DEBUG: adam: grad:      ", self._grad.cpu().numpy())
    #print("DEBUG: adam: param pre: ", self._param.cpu().numpy())
    self._mavg_grad_1.add_(decay_rate_1 * (self._grad - self._mavg_grad_1))
    self._mavg_grad_2.add_(decay_rate_2 * (torch.mul(self._grad, self._grad) - self._mavg_grad_2))
    self._param.add_(-step_size * (norm_1 * self._mavg_grad_1) / ((norm_2 * self._mavg_grad_2).sqrt() + epsilon))
    #print("DEBUG: adam: param post:", self._param.cpu().numpy())

    assert self._dim == deserialize_vars(param_vars, self._param)

    self._iter_ct += 1

    return self._loss

if __name__ == "__main__":
  import torch.cuda

  dtype = torch.cuda.FloatTensor

  x = var(torch.ones(4).type(dtype))
  batch_loss_fn = lambda _: 0.5 * x.dot(x)

  #sgd = SGDOptimizer({
  sgd = AdamOptimizer({
    "batch_size": 1,
    "minibatch_size": 1,
    "step_size": 0.1,
    "decay_rate_1": 0.1,
    "decay_rate_2": 0.001,
    "epsilon": 1.0e-8,
  })

  sgd.reset([x], dtype)
  print("DEBUG: loss:", batch_loss_fn(1).data.cpu().numpy())
  print("DEBUG: param:", x.data.cpu().numpy())
  for _ in range(10):
    sgd.step([x], batch_loss_fn)
    print("DEBUG: loss dim:", batch_loss_fn(1).data.size())
    print("DEBUG: loss:", batch_loss_fn(1).data.cpu().numpy())
    print("DEBUG: param:", x.data.cpu().numpy())
