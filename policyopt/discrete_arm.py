#!/usr/bin/env python3.5

from policyopt.autodiff import *
from policyopt.diffopt import *
from policyopt.policy import *
from policyopt.sample import *

import gym
import numpy as np
import torch
import torch.cuda
import torch.nn.functional

#import gc
import time

def build_lstsq_v_loss_fn(v_fn):
  def v_loss_fn(obs, target):
    v_val = torch.squeeze(v_fn(obs), 1)
    residual = v_val - target
    sq_loss = 0.5 * residual * residual
    return torch.sum(sq_loss)
  return v_loss_fn

def build_huber_v_loss_fn(v_fn):
  def v_loss_fn(obs, target):
    v_val = torch.squeeze(v_fn(obs), 1)
    hu_loss = torch.nn.functional.smooth_l1_loss(v_val, target)
    return torch.sum(hu_loss)
  return v_loss_fn

def build_lstsq_q_loss_fn(q_fn):
  def q_loss_fn(obs, act_idx, target):
    q_val = q_fn(obs)
    q_act_val = torch.squeeze(torch.gather(q_val, 1, torch.unsqueeze(act_idx, 1)), 1)
    residual = q_act_val - target
    sq_loss = 0.5 * residual * residual
    return torch.sum(sq_loss)
  return q_loss_fn

def build_huber_q_loss_fn(q_fn):
  def q_loss_fn(obs, act_idx, target):
    q_val = q_fn(obs)
    q_act_val = torch.squeeze(torch.gather(q_val, 1, torch.unsqueeze(act_idx, 1)), 1)
    hu_loss = torch.nn.functional.smooth_l1_loss(q_act_val, target)
    return torch.sum(hu_loss)
  return q_loss_fn

class DiscreteARMPolicy(object):
  def __init__(self, action_dim, ccq_fn, v_fn):
    self._act_dim = action_dim
    self._ccq_fn = ccq_fn
    self._v_fn = v_fn

  def __call__(self, obs_buf, cond_act_idx=None):
    if isinstance(obs_buf, np.ndarray):
      obs = const_var(torch.unsqueeze(torch.from_numpy(obs_buf), 0))
    elif isinstance(obs_buf, torch.cuda.ByteTensor):
      obs = const_var(obs_buf)
      #print("DEBUG: torch obs: size:", obs.size())
    else:
      # TODO
      print("WARNING: arm policy: unknown type:", type(obs_buf))
      raise NotImplementedError
    #regrets_var = torch.clamp((torch.squeeze(self._ccq_fn(obs) - self._v_fn(obs))), min=0.0)
    regrets_var = torch.clamp(self._ccq_fn(obs) - self._v_fn(obs), min=0.0)
    regrets = regrets_var.data.cpu().numpy()
    sum_regrets = np.sum(regrets, axis=1, keepdims=True)
    act_dist = regrets / sum_regrets
    assert len(act_dist.shape) == 2
    if cond_act_idx is not None:
      act_probs = []
      for idx in range(act_dist.shape[0]):
        if sum_regrets[idx,0] > 0.0:
          act_prob = act_dist[idx,int(cond_act_idx[idx])]
        else:
          act_prob = np.float32(1.0 / float(self._act_dim))
        act_probs.append(act_prob)
      return act_probs
    else:
      act_idxs = []
      act_probs = []
      for idx in range(act_dist.shape[0]):
        if sum_regrets[idx,0] > 0.0:
          act_idx = np.random.choice(self._act_dim, p=act_dist[idx,:])
          act_prob = act_dist[idx,act_idx]
        else:
          act_idx = np.random.choice(self._act_dim)
          act_prob = np.float32(1.0 / float(self._act_dim))
        act_idxs.append(act_idx)
        act_probs.append(act_prob)
      return act_idxs, act_probs

class DiscreteARM(object):
  def __init__(self, cfg, batch_cfg, eval_cfg, opt_cfg):
    self._cfg = cfg.copy()
    self._batch_cfg = batch_cfg.copy()
    self._eval_cfg = eval_cfg.copy()
    self._opt_cfg = opt_cfg.copy()
    self._iter_ct = 0
    self._step_ct = 0
    self._batch_cache = SampleBatchCache(self._cfg["num_cached_batches"])

    # These are the vanilla ARM losses and parameters.
    self._prev_v_param_vars = None
    self._prev_v_fn = None
    self._prev_ccq_param_vars = None
    self._prev_ccq_fn = None
    self._tg_v_param_vars = None
    self._tg_v_fn = None
    self._v_opt = None
    self._v_loss_fn = None
    self._v_param_vars = None
    self._v_fn = None
    self._ccq_opt = None
    self._ccq_loss_fn = None
    self._ccq_param_vars = None
    self._ccq_fn = None

    self._init_policy = None
    self._policy = None

  def reset(self, env, prev_v_param_vars, prev_v_fn, prev_ccq_param_vars, prev_ccq_fn, tg_v_param_vars, tg_v_fn, v_param_vars, v_fn, ccq_param_vars, ccq_fn):
    self._iter_ct = 0
    self._step_ct = 0
    self._batch_cache.reset()

    self._prev_v_param_vars = prev_v_param_vars
    self._prev_v_fn = prev_v_fn
    self._prev_ccq_param_vars = prev_ccq_param_vars
    self._prev_ccq_fn = prev_ccq_fn

    self._tg_v_param_vars = tg_v_param_vars
    self._tg_v_fn = tg_v_fn

    self._v_opt = AdamOptimizer(self._opt_cfg)
    #self._v_loss_fn = build_lstsq_v_loss_fn(v_fn)
    self._v_loss_fn = build_huber_v_loss_fn(v_fn)
    self._v_param_vars = v_param_vars
    self._v_fn = v_fn
    self._v_opt.reset(self._v_param_vars, dtype=torch.cuda.FloatTensor)

    self._ccq_opt = AdamOptimizer(self._opt_cfg)
    #self._ccq_loss_fn = build_lstsq_q_loss_fn(ccq_fn)
    self._ccq_loss_fn = build_huber_q_loss_fn(ccq_fn)
    self._ccq_param_vars = ccq_param_vars
    self._ccq_fn = ccq_fn
    self._ccq_opt.reset(self._ccq_param_vars, dtype=torch.cuda.FloatTensor)

    self._init_policy = UniformCategoricalPolicy(env.action_space.n)
    self._policy = DiscreteARMPolicy(env.action_space.n, self._ccq_fn, self._v_fn)

  def step(self, env):
    print("DEBUG: arm: iteration: {} total steps: {}".format(self._iter_ct, self._step_ct))

    #self._batch_cache.drop_fifo(self._cfg["cache_size"])
    online_batch = SampleBatch()
    if self._iter_ct == 0: # and self._cfg["initialize_uniform"]:
      online_batch.resample_categorical(self._batch_cfg, env, self._init_policy)
    else:
      online_batch.resample_categorical(self._batch_cfg, env, self._policy)
    self._batch_cache.append(online_batch)

    self._batch_cache.vectorize_categorical()
    if self._iter_ct == 0: # and self._cfg["initialize_uniform"]:
      #self._batch_cache.reweight_categorical(self._init_policy)
      pass
    else:
      self._batch_cache.reweight_categorical(self._policy)

    # TODO: run ARM optimization.

    nsteps = self._cfg["nsteps"]
    discount = self._cfg["discount"]
    nstep_discount = float(np.power(discount, nsteps))
    res_scale = None
    if "res_scale" in self._cfg:
      res_scale = self._cfg["res_scale"]
    #if res_scale is None:
    #  res_scale = 1.0

    initial_num_arm_iters = self._cfg["initial_num_arm_iters"]
    num_arm_iters = self._cfg["num_arm_iters"]

    minibatch_size = self._cfg["minibatch_size"]
    arm_target_step_size = self._cfg["arm_target_step_size"]

    #self._v_opt.reset(self._v_param_vars, dtype=torch.cuda.FloatTensor)
    #self._ccq_opt.reset(self._ccq_param_vars, dtype=torch.cuda.FloatTensor)
    copy_vars(self._tg_v_param_vars, self._v_param_vars)

    if self._iter_ct == 0: # and self._cfg["initialize_uniform"]:
      curr_num_arm_iters = initial_num_arm_iters
    else:
      curr_num_arm_iters = num_arm_iters

    avg_v_loss = torch.zeros(1).cuda()
    avg_ccq_loss = torch.zeros(1).cuda()
    last_display_iter = 0
    last_t = time.perf_counter()

    for t in range(curr_num_arm_iters):
      # TODO
      #idxs, nstep_idxs, act_idx, nres, done = sample_minibatch_indexes(minibatch_size, nsteps, discount, res_scale)
      idxs, nstep_idxs, obs_ks, obs_kpns, act_idx_ks, v_rets, q_rets, dones = self._batch_cache.sample_minibatch(minibatch_size, nsteps=nsteps, discount=discount, res_scale=res_scale, categorical=True)

      #obs_k = const_var(obs_buffer[idxs,:])
      #obs_kpn = const_var(obs_buffer[nstep_idxs,:])
      #act_idx_k = const_var(act_idx).cuda()
      #nres_k = const_var(nres).cuda()
      #done_kpn = const_var(done).cuda()
      #print("DEBUG: step: idxs:", idxs)
      #obs_k = const_var(self._batch_cache.obs_buffer[idxs,:])
      #obs_kpn = const_var(self._batch_cache.obs_buffer[nstep_idxs,:])
      #act_idx_k = const_var(self._batch_cache.act_idx_buffer[idxs,]).cuda()
      obs_k = const_var(obs_ks)
      obs_kpn = const_var(obs_kpns)
      act_idx_k = const_var(act_idx_ks).cuda()
      v_ret_k = const_var(v_rets).cuda()
      q_ret_k = const_var(q_rets).cuda()
      done_kpn = const_var(dones).cuda()
      tg_v_kpn = self._tg_v_fn(obs_kpn)
      #tg_v_kpn = torch.mul((1.0 - done_kpn), torch.squeeze(tg_v_kpn, 1))
      tg_v_kpn = (1.0 - done_kpn) * torch.squeeze(tg_v_kpn, 1)
      target_v_k = v_ret_k + nstep_discount * tg_v_kpn
      if self._iter_ct == 0: # and self._cfg["initialize_uniform"]:
        target_ccq_k = q_ret_k + nstep_discount * tg_v_kpn
      else:
        target_ccq_k = torch.clamp((torch.squeeze(torch.gather(self._prev_ccq_fn(obs_k), 1, torch.unsqueeze(act_idx_k, 1)) - self._prev_v_fn(obs_k), 1)), min=0.0) + q_ret_k + nstep_discount * tg_v_kpn
      target_v_k = target_v_k.detach()
      target_ccq_k = target_ccq_k.detach()
      #print("DEBUG: target ccq:", target_ccq_k.size())

      v_loss = self._v_opt.step(self._v_param_vars, lambda _: self._v_loss_fn(obs_k, target_v_k))
      ccq_loss = self._ccq_opt.step(self._ccq_param_vars, lambda _: self._ccq_loss_fn(obs_k, act_idx_k, target_ccq_k))
      average_vars(arm_target_step_size, self._tg_v_param_vars, self._v_param_vars)

      avg_v_loss.add_(v_loss)
      avg_ccq_loss.add_(ccq_loss)

      if (t+1) % 400 == 0 or (t+1) == curr_num_arm_iters:
        elapsed_display_iters = float(t + 1 - last_display_iter)
        lap_t = time.perf_counter()
        elapsed_s = float(lap_t - last_t)
        print("DEBUG: arm:   iters: {} v loss: {:.6f} ccq loss: {:.6f} elapsed: {:.3f}".format(
            t+1,
            float(avg_v_loss.cpu().numpy()) / elapsed_display_iters,
            float(avg_ccq_loss.cpu().numpy()) / elapsed_display_iters,
            float(elapsed_s),
        ))
        avg_v_loss.zero_()
        avg_ccq_loss.zero_()
        last_display_iter = t + 1
        last_t = lap_t

      #gc.collect()

    # TODO: cleanup.

    #del obs_buffer

    #del obs_buffer_h
    #del act_idx_buffer_h
    #del res_buffer_h
    #del done_buffer_h

    # TODO: update state.

    self._iter_ct += 1
    self._step_ct += online_batch.step_count()
    copy_vars(self._prev_v_param_vars, self._v_param_vars)
    copy_vars(self._prev_ccq_param_vars, self._ccq_param_vars)

    return online_batch.step_count()
    #return online_batch.step_count(), online_batch.traj_count()

  def eval(self, env):
    # TODO
    raise NotImplementedError

if __name__ == "__main__":
  pass
