from policyopt.autodiff import *
from policyopt.utils import perf_counter

import numpy as np
import torch
import torch.cuda

from collections import deque

class SampleTransition(object):
  def __init__(self):
    self.action = None
    self.action_index = None   # discrete actions only.
    self.action_prob = None
    self.response = None
    self.terminal = None
    self.est_adv = None
    self.est_nadv = None
    self.next_obs = None
    self.next_est_val = None

class SampleTraj(object):
  def __init__(self):
    self.init_obs = None
    self.init_est_val = None
    self.steps = []

  def step_count(self):
    return len(self.steps)

  def sum_return(self):
    r = 0.0
    for transition in self.steps:
      r += transition.response
    return r

  def reset(self):
    self.init_obs = None
    self.steps[:] = []

  def init(self, init_obs):
    self.init_obs = init_obs

  def append(self, action, response, terminal, next_obs, action_prob=None):
    transition = SampleTransition()
    transition.action = action
    transition.action_prob = action_prob
    transition.response = response
    transition.terminal = terminal
    transition.next_obs = next_obs
    self.steps.append(transition)

  def append_categorical(self, action_index, action_prob, response, terminal, next_obs):
    transition = SampleTransition()
    transition.action_index = action_index
    transition.action_prob = action_prob
    transition.response = response
    transition.terminal = terminal
    transition.next_obs = next_obs
    self.steps.append(transition)

  def curr_obs(self):
    nsteps = len(self.steps)
    if nsteps == 0:
      return self.init_obs
    else:
      return self.steps[-1].next_obs

  def prev_obs(self):
    nsteps = len(self.steps)
    if nsteps == 0:
      return None
    elif nsteps == 1:
      return self.init_obs
    else:
      return self.steps[-2].next_obs

class SampleBatch(object):
  def __init__(self, prefix=None):
    self._prefix = prefix
    self._step_ct = 0
    self.trajs = []

    self.obs_buffer = None
    self.act_buffer = None
    self.act_idx_buffer = None
    self.act_prob_buffer = None
    self.res_buffer = None
    self.end_buffer = None
    self.done_buffer = None

    self.val_buffer = None
    self.adv_buffer = None
    self.nadv_buffer = None
    self.weight_buffer = None

  def step_count(self):
    return self._step_ct

  def traj_count(self):
    return len(self.trajs)

  def resample(self, cfg, env, policy):
    if self._prefix is not None:
      label = "{} batch".format(self._prefix)
    else:
      label = "batch"
    print("DEBUG: {}: resampling policy...".format(label))
    batch_start_t = perf_counter()
    self._step_ct = 0
    self.trajs[:] = []
    avg_return = 0.0
    max_return = -float("inf")
    min_return = float("inf")
    while self._step_ct < cfg["sample_size"]:
      ep_start_t = perf_counter()
      traj = SampleTraj()
      obs = env.reset()
      traj.init(obs)
      #for k in range(cfg["max_horizon"]):
      while True:
        action, action_prob = policy(obs)
        action = action[0]
        action_prob = action_prob[0]
        obs, response, done, _ = env.step(action)
        traj.append(action, action_prob, response, done, obs)
        if done:
          break
      self._step_ct += traj.step_count()
      self.trajs.append(traj)
      avg_return += 1.0 / float(len(self.trajs)) * (traj.sum_return() - avg_return)
      max_return = max(max_return, traj.sum_return())
      min_return = min(min_return, traj.sum_return())
      ep_lap_t = perf_counter()
      ep_elapsed_s = float(ep_lap_t - ep_start_t)
      elapsed_s = float(ep_lap_t - batch_start_t)
      print("DEBUG: {}:   trajs: {} ret: {:.3f} steps: {} batch steps: {} elapsed: {:.3f} batch elapsed: {:.3f}".format(
          label, len(self.trajs), traj.sum_return(), traj.step_count(), self._step_ct, ep_elapsed_s, elapsed_s))
    #elapsed_s = float(perf_counter() - batch_start_t)
    print("DEBUG: {}: trajs: {} steps: {} avg ret: {:.3f} max ret: {:.3f} min ret: {:.3f} elapsed: {:.3f}".format(
        label, self.traj_count(), self.step_count(), avg_return, max_return, min_return, elapsed_s))

  def resample_categorical(self, cfg, env, cat_policy):
    print("DEBUG: batch: resampling categorical policy...")
    batch_start_t = perf_counter()
    self._step_ct = 0
    self.trajs[:] = []
    avg_return = 0.0
    max_return = -float("inf")
    min_return = float("inf")
    while self._step_ct < cfg["sample_size"]:
      ep_start_t = perf_counter()
      traj = SampleTraj()
      obs = env.reset()
      traj.init(obs)
      #for k in range(cfg["max_horizon"]):
      while True:
        action_index, action_prob = cat_policy(obs)
        action_index = action_index[0]
        action_prob = action_prob[0]
        obs, response, done, _ = env.step(action_index)
        traj.append_categorical(action_index, action_prob, response, done, obs)
        if done:
          break
      self._step_ct += traj.step_count()
      self.trajs.append(traj)
      avg_return += 1.0 / float(len(self.trajs)) * (traj.sum_return() - avg_return)
      max_return = max(max_return, traj.sum_return())
      min_return = min(min_return, traj.sum_return())
      ep_lap_t = perf_counter()
      ep_elapsed_s = float(ep_lap_t - ep_start_t)
      elapsed_s = float(ep_lap_t - batch_start_t)
      print("DEBUG: batch:   trajs: {} ret: {:.3f} steps: {} batch steps: {} elapsed: {:.3f} batch elapsed: {:.3f}".format(
          len(self.trajs), traj.sum_return(), traj.step_count(), self._step_ct, ep_elapsed_s, elapsed_s))
    #elapsed_s = float(perf_counter() - batch_start_t)
    print("DEBUG: batch: trajs: {} steps: {} avg ret: {:.3f} max ret: {:.3f} min ret: {:.3f} elapsed: {:.3f}".format(
        self.traj_count(), self.step_count(), avg_return, max_return, min_return, elapsed_s))

  def old_resample(self, cfg, env, policy):
    self._step_ct = 0
    self.trajs[:] = []
    while self._step_ct < cfg["sample_size"]:
      traj = SampleTraj()
      obs = env.reset()
      traj.init(obs)
      for k in range(cfg["max_horizon"]):
        action, action_prob = policy(obs)
        obs, response, done, _ = env.step(action)
        traj.append(action, action_prob, response, done, obs)
        if done:
          break
      self._step_ct += traj.step_count()
      self.trajs.append(traj)

  def add_entropy_res(self, entropy_scale, entropy_fn, chunk_size=64):
    for traj in self.trajs:
      chunk_k = []
      chunk_obs = []
      chunk_act = []
      for k in range(traj.step_count()):
        # TODO
        if len(chunk_obs) == chunk_size:
          obs = const_var(torch.from_numpy(np.array(chunk_obs))).cuda()
          act = const_var(torch.from_numpy(np.array(chunk_act))).cuda()
          chunk_ent = entropy_fn(obs, act)
          chunk_ent = chunk_ent.data.cpu().numpy()
          for j in range(len(chunk_ent)):
            kp = chunk_k[j]
            traj.steps[kp].response += entropy_scale * float(chunk_ent[j])
          chunk_k[:] = []
          chunk_obs[:] = []
          chunk_act[:] = []
        chunk_k.append(k)
        chunk_act.append(traj.steps[k].action)
        if k == 0:
          chunk_obs.append(traj.init_obs)
        else:
          chunk_obs.append(traj.steps[k-1].next_obs)
      if len(chunk_obs) > 0:
        obs = const_var(torch.from_numpy(np.array(chunk_obs))).cuda()
        act = const_var(torch.from_numpy(np.array(chunk_act))).cuda()
        chunk_ent = entropy_fn(obs, act)
        chunk_ent = chunk_ent.data.cpu().numpy()
        for j in range(len(chunk_ent)):
          kp = chunk_k[j]
          traj.steps[kp].response += entropy_scale * float(chunk_ent[j])

  def evaluate_value_fn(self, v_fn, chunk_size=64):
    for traj in self.trajs:
      chunk_k = []
      chunk_obs = []
      chunk_k.append(-1)
      chunk_obs.append(traj.init_obs)
      for k in reversed(range(traj.step_count())):
        if len(chunk_obs) == chunk_size:
          chunk_v = v_fn(const_var(torch.from_numpy(np.array(chunk_obs))))
          chunk_v = chunk_v.data.cpu().numpy()
          for j in range(len(chunk_v)):
            kp = chunk_k[j]
            if kp == -1:
              traj.init_est_val = float(chunk_v[j])
            else:
              traj.steps[kp].next_est_val = float(chunk_v[j])
          chunk_k[:] = []
          chunk_obs[:] = []
        chunk_k.append(k)
        chunk_obs.append(traj.steps[k].next_obs)
      if len(chunk_obs) > 0:
        chunk_v = v_fn(const_var(torch.from_numpy(np.array(chunk_obs))))
        chunk_v = chunk_v.data.cpu().numpy()
        for j in range(len(chunk_v)):
          kp = chunk_k[j]
          if kp == -1:
            traj.init_est_val = float(chunk_v[j])
          else:
            traj.steps[kp].next_est_val = float(chunk_v[j])

  def calculate_lambda_advs(self, discount=None, lambda_=None):
    all_est_advs = []
    for traj in self.trajs:
      est_adv = 0.0
      for k in reversed(range(traj.step_count())):
        r_k = traj.steps[k].response
        if k == 0:
          v_k = traj.init_est_val
        else:
          v_k = traj.steps[k-1].next_est_val
        v_kp1 = traj.steps[k].next_est_val
        done_kp1 = 1.0 if traj.steps[k].terminal else 0.0
        delta_k = r_k + discount * (1.0 - done_kp1) * v_kp1 - v_k
        est_adv = delta_k + lambda_ * discount * (1.0 - done_kp1) * est_adv
        all_est_advs.append(est_adv)
        traj.steps[k].est_adv = est_adv
    all_est_advs = np.array(all_est_advs)
    est_adv_mean = np.mean(all_est_advs)
    est_adv_std = np.std(all_est_advs)
    for traj in self.trajs:
      for k in range(traj.step_count()):
        traj.steps[k].est_nadv = (traj.steps[k].est_adv - est_adv_mean) / est_adv_std
    return float(est_adv_mean), float(est_adv_std)

  def calculate_gaussian_kl(self, d, old_pol_dist_fn, new_pol_dist_fn, chunk_size=64):
    def calculate_gaussian_kl_torch(old_mean, old_logstd, new_mean, new_logstd):
      return torch.sum((new_logstd - old_logstd) + 0.5 * (torch.exp(old_logstd) ** 2 + (new_mean - old_mean) ** 2) / (torch.exp(new_logstd) ** 2), dim=1) - 0.5 * float(d)
    all_kl = []
    for traj in self.trajs:
      chunk_obs = []
      chunk_obs.append(traj.init_obs)
      for k in range(traj.step_count()):
        if len(chunk_obs) == chunk_size:
          old_mean, old_logstd = old_pol_dist_fn(const_var(torch.from_numpy(np.array(chunk_obs))))
          new_mean, new_logstd = new_pol_dist_fn(const_var(torch.from_numpy(np.array(chunk_obs))))
          chunk_kl = calculate_gaussian_kl_torch(old_mean, old_logstd, new_mean, new_logstd)
          chunk_kl = chunk_kl.data.cpu().numpy()
          for j in range(len(chunk_kl)):
            all_kl.append(chunk_kl[j])
          chunk_obs[:] = []
        chunk_obs.append(traj.steps[k].next_obs)
      if len(chunk_obs) > 0:
        old_mean, old_logstd = old_pol_dist_fn(const_var(torch.from_numpy(np.array(chunk_obs))))
        new_mean, new_logstd = new_pol_dist_fn(const_var(torch.from_numpy(np.array(chunk_obs))))
        chunk_kl = calculate_gaussian_kl_torch(old_mean, old_logstd, new_mean, new_logstd)
        chunk_kl = chunk_kl.data.cpu().numpy()
        for j in range(len(chunk_kl)):
          all_kl.append(chunk_kl[j])
    all_kl = np.array(all_kl)
    return float(np.mean(all_kl)), float(np.std(all_kl))

  def is_vectorized(self):
    return self.obs_buffer is not None

  def vectorize(self, action_probs=False, mc_values=False, est_val_adv=False, discount=None):
    xbatch_size = self.step_count() + self.traj_count()

    obs_buffer_h = torch.FloatTensor(*tuple([xbatch_size] + list(self.trajs[0].init_obs.shape)))
    act_buffer_h = torch.FloatTensor(*tuple([xbatch_size] + list(self.trajs[0].steps[0].action.shape)))
    if action_probs:
      act_prob_buffer_h = torch.FloatTensor(xbatch_size)
    res_buffer_h = torch.FloatTensor(xbatch_size)
    if mc_values:
      val_buffer_h = torch.FloatTensor(xbatch_size)
    if est_val_adv:
      val_buffer_h = torch.FloatTensor(xbatch_size)
      adv_buffer_h = torch.FloatTensor(xbatch_size)
      nadv_buffer_h = torch.FloatTensor(xbatch_size)
    end_buffer_h = torch.FloatTensor(xbatch_size)
    done_buffer_h = torch.FloatTensor(xbatch_size)

    obs_buffer_h.zero_()
    act_buffer_h.zero_()
    if action_probs:
      act_prob_buffer_h.zero_()
    res_buffer_h.zero_()
    if mc_values:
      val_buffer_h.zero_()
    if est_val_adv:
      val_buffer_h.zero_()
      adv_buffer_h.zero_()
      nadv_buffer_h.zero_()
    end_buffer_h.zero_()
    done_buffer_h.zero_()

    traj_offsets = []
    traj_lengths = []
    traj_offset_ctr = 0
    for i, traj in enumerate(self.trajs):
      traj_offsets.append(traj_offset_ctr)
      traj_lengths.append(traj.step_count())
      traj_term = False
      obs_buffer_h[traj_offsets[i],:].copy_(torch.from_numpy(traj.init_obs))
      done_buffer_h[traj_offsets[i]] = 0.0
      for k in range(traj.step_count()):
        if traj.steps[k].terminal:
          traj_term = True
        if traj_term:
          assert traj.steps[k].terminal
        act_buffer_h[traj_offsets[i] + k,:].copy_(torch.from_numpy(traj.steps[k].action))
        if action_probs:
          act_prob_buffer_h[traj_offsets[i] + k] = float(traj.steps[k].action_prob)
        res_buffer_h[traj_offsets[i] + k] = traj.steps[k].response
        done_buffer_h[traj_offsets[i] + k + 1] = 1.0 if traj.steps[k].terminal else 0.0
        obs_buffer_h[traj_offsets[i] + k + 1,:].copy_(torch.from_numpy(traj.steps[k].next_obs))
        if est_val_adv:
          for k in reversed(range(traj.step_count())):
            if k == 0:
              val_buffer_h[traj_offsets[i] + k] = traj.init_est_val
            else:
              val_buffer_h[traj_offsets[i] + k] = traj.steps[k-1].next_est_val
            adv_buffer_h[traj_offsets[i] + k] = traj.steps[k].est_adv
            nadv_buffer_h[traj_offsets[i] + k] = traj.steps[k].est_nadv
      act_buffer_h[traj_offsets[i] + traj.step_count(),:].copy_(torch.from_numpy(np.zeros(traj.steps[k].action.shape)))
      if action_probs:
        act_prob_buffer_h[traj_offsets[i] + traj.step_count()] = 0.0
      res_buffer_h[traj_offsets[i] + traj.step_count()] = 0.0
      end_buffer_h[traj_offsets[i] + traj.step_count()] = 1.0
      if mc_values:
        assert discount is not None
        mc_val = 0.0
        for k in reversed(range(traj.step_count())):
          mc_val = float(res_buffer_h[k]) + discount * mc_val
          val_buffer_h[traj_offsets[i] + k] = mc_val
      traj_offset_ctr += traj.step_count() + 1
    assert traj_offset_ctr == xbatch_size

    obs_buffer = torch.cuda.FloatTensor(*tuple([xbatch_size] + list(self.trajs[0].init_obs.shape)))
    act_buffer = torch.cuda.FloatTensor(*tuple([xbatch_size] + list(self.trajs[0].steps[0].action.shape)))
    obs_buffer.copy_(obs_buffer_h)
    act_buffer.copy_(act_buffer_h)

    torch.cuda.synchronize()

    print("DEBUG: obs buffer size: {}".format(obs_buffer.size()))
    print("DEBUG: act buffer size: {}".format(act_buffer.size()))

    del obs_buffer_h
    del act_buffer_h

    self.xbatch_size = xbatch_size
    self.obs_buffer = obs_buffer
    self.act_buffer = act_buffer
    if action_probs:
      self.act_prob_buffer = act_prob_buffer_h
    self.res_buffer = res_buffer_h
    if mc_values:
      self.val_buffer = val_buffer_h
    if est_val_adv:
      self.val_buffer = val_buffer_h
      self.adv_buffer = adv_buffer_h
      self.nadv_buffer = nadv_buffer_h
    self.end_buffer = end_buffer_h
    self.done_buffer = done_buffer_h

  def vectorize_categorical(self):
    xbatch_size = self.step_count() + self.traj_count()

    obs_buffer_h = torch.ByteTensor(*tuple([xbatch_size] + list(self.trajs[0].init_obs.shape)))
    act_idx_buffer_h = torch.LongTensor(xbatch_size)
    act_prob_buffer_h = torch.FloatTensor(xbatch_size)
    res_buffer_h = torch.FloatTensor(xbatch_size)
    end_buffer_h = torch.FloatTensor(xbatch_size)
    done_buffer_h = torch.FloatTensor(xbatch_size)

    obs_buffer_h.zero_()
    act_idx_buffer_h.zero_()
    act_prob_buffer_h.zero_()
    res_buffer_h.zero_()
    end_buffer_h.zero_()
    done_buffer_h.zero_()

    traj_offsets = []
    traj_lengths = []
    traj_offset_ctr = 0
    for i, traj in enumerate(self.trajs):
      traj_offsets.append(traj_offset_ctr)
      traj_lengths.append(traj.step_count())
      traj_term = False
      obs_buffer_h[traj_offsets[i],:].copy_(torch.from_numpy(traj.init_obs))
      done_buffer_h[traj_offsets[i]] = 0.0
      for k in range(traj.step_count()):
        if traj.steps[k].terminal:
          traj_term = True
        if traj_term:
          assert traj.steps[k].terminal
        act_idx_buffer_h[traj_offsets[i] + k] = traj.steps[k].action_index
        act_prob_buffer_h[traj_offsets[i] + k] = float(traj.steps[k].action_prob)
        res_buffer_h[traj_offsets[i] + k] = traj.steps[k].response
        done_buffer_h[traj_offsets[i] + k + 1] = 1.0 if traj.steps[k].terminal else 0.0
        obs_buffer_h[traj_offsets[i] + k + 1].copy_(torch.from_numpy(traj.steps[k].next_obs))
      act_idx_buffer_h[traj_offsets[i] + traj.step_count()] = -1
      act_prob_buffer_h[traj_offsets[i] + traj.step_count()] = 0.0
      res_buffer_h[traj_offsets[i] + traj.step_count()] = 0.0
      end_buffer_h[traj_offsets[i] + traj.step_count()] = 1.0
      traj_offset_ctr += traj.step_count() + 1
    assert traj_offset_ctr == xbatch_size

    obs_buffer_d = torch.cuda.ByteTensor(*tuple([xbatch_size] + list(self.trajs[0].init_obs.shape)))
    obs_buffer_d.copy_(obs_buffer_h)
    torch.cuda.synchronize()
    del obs_buffer_h

    self.xbatch_size = xbatch_size
    #self.obs_buffer = obs_buffer_h
    self.obs_buffer = obs_buffer_d
    self.act_idx_buffer = act_idx_buffer_h
    self.act_prob_buffer = act_prob_buffer_h
    self.res_buffer = res_buffer_h
    self.end_buffer = end_buffer_h
    self.done_buffer = done_buffer_h

    print("DEBUG: obs buffer size: {}".format(self.obs_buffer.size()))
    print("DEBUG: act idx buf size: {}".format(self.act_idx_buffer.size()))

  def reweight_categorical(self, online_policy, clip_weight=1.0, chunk_size=32):
    weight_buffer_h = torch.FloatTensor(self.xbatch_size)

    num_chunks = (self.xbatch_size + chunk_size - 1) // chunk_size
    for chunk_idx in range(num_chunks):
      idxs = []
      num_real_idxs = 0
      for i in range(chunk_size):
        idx = i + chunk_size * chunk_idx
        if idx >= self.xbatch_size:
          idxs.append(self.xbatch_size - 1)
        else:
          idxs.append(idx)
          num_real_idxs += 1
      #print("DEBUG: reweight: idxs:", idxs)
      #print("DEBUG: reweight: obs buffer: type:", type(self.obs_buffer))
      act_prob = online_policy(self.obs_buffer[idxs,:], cond_act_idx=self.act_idx_buffer[idxs,])
      for i in range(num_real_idxs):
        idx = idxs[i]
        if self.done_buffer[idx] != 0.0 or self.end_buffer[idx] != 0.0:
          weight_buffer_h[idx] = 0.0
        else:
          #w = torch.clamp(act_prob[i] / self.act_prob_buffer[idx], min=0.0, max=clip_weight)
          w = np.clip(act_prob[i] / float(self.act_prob_buffer[idx]), 0.0, clip_weight)
          weight_buffer_h[idx] = w

    if self.weight_buffer is not None:
      del self.weight_buffer
      self.weight_buffer = None

    self.weight_buffer = weight_buffer_h

  def cleanup(self):
    if self.obs_buffer is not None:
      del self.obs_buffer
      self.obs_buffer = None
    if self.act_buffer is not None:
      del self.act_buffer
      self.act_buffer = None
    if self.act_idx_buffer is not None:
      del self.act_idx_buffer
      self.act_idx_buffer = None
    if self.act_prob_buffer is not None:
      del self.act_prob_buffer
      self.act_prob_buffer = None
    if self.res_buffer is not None:
      del self.res_buffer
      self.res_buffer = None
    if self.end_buffer is not None:
      del self.end_buffer
      self.end_buffer = None
    if self.done_buffer is not None:
      del self.done_buffer
      self.done_buffer = None
    if self.weight_buffer is not None:
      del self.weight_buffer
      self.weight_buffer = None

  def sample_transition(self, nsteps=1, discount=1.0, res_scale=None, res_weight=True, strategy="uniform"):
    if strategy == "uniform":
      idx = None
      while True:
        idx = int(np.random.choice(self.xbatch_size))
        if self.done_buffer[idx] != 0.0 or self.end_buffer[idx] != 0.0:
          continue
        break
      assert idx is not None

      _done_idx = None
      _end_idx = None
      for step in range(1, nsteps + 1):
        if _done_idx is None:
          if self.done_buffer[idx+step] != 0.0:
            _done_idx = idx + step
        if self.end_buffer[idx+step] != 0.0:
          _end_idx = idx + step
          break

      nstep_idx = None
      if _done_idx is None and _end_idx is None:
        nstep_idx = idx + nsteps
      elif _done_idx is not None:
        # NOTE: Check the `done` case before the `end` case.
        assert _done_idx >= idx
        assert _done_idx <= idx + nsteps
        nstep_idx = _done_idx
      elif _end_idx is not None:
        assert _end_idx >= idx
        assert _end_idx <= idx + nsteps
        nstep_idx = _end_idx
      else:
        raise NotImplementedError
      assert nstep_idx is not None
      assert nstep_idx >= idx
      assert nstep_idx <= idx + nsteps

      v_nres = 0.0
      q_nres = 0.0
      for step in reversed(range(nsteps)):
        if _end_idx is not None:
          if _end_idx <= idx + step:
            continue
        if _done_idx is not None:
          if _done_idx <= idx + step:
            continue
        if res_weight:
          w = float(self.weight_buffer[idx+step])
        else:
          w = 1.0
        r = float(self.res_buffer[idx+step])
        v_nres = w * (r + discount * v_nres)
        if step != 0:
          q_nres = w * (r + discount * q_nres)
        else:
          q_nres = r + w * discount * q_nres
      if res_scale is not None:
        v_nres *= res_scale
        q_nres *= res_scale

      done = 0.0
      if _done_idx is not None:
        done = 1.0

      return idx, nstep_idx, v_nres, q_nres, done

    else:
      # TODO
      raise NotImplementedError

class SampleBatchCache(object):
  def __init__(self, max_num_batches=None):
    self._max_num_batches = max_num_batches
    self._step_ct = 0
    self._traj_ct = 0
    self.batches = deque([], maxlen=self._max_num_batches)
    self.online_batch_idx = None
    self.xbatch_sizes = None
    self.xbatch_offsets = None

  def step_count(self):
    return self._step_ct

  def traj_count(self):
    return self._traj_ct

  def batch_count(self):
    return len(self.batches)

  def reset(self):
    self._step_ct = 0
    self._traj_ct = 0
    self.batches.clear()
    self.online_batch_idx = None
    self.xbatch_sizes = None
    self.xbatch_offsets = None

  def flush(self):
    while len(self.batches) > 0:
      print("DEBUG: batch cache: drop: trajs: {} steps: {}".format(self.batches[0].traj_count(), self.batches[0].step_count()))
      pop_batch = self.batches.popleft()
      self._step_ct -= pop_batch.step_count()
      self._traj_ct -= pop_batch.traj_count()
      assert self._step_ct >= 0
      assert self._traj_ct >= 0
      pop_batch.cleanup()
    assert self._step_ct == 0
    assert self._traj_ct == 0

  def append(self, new_batch):
    while len(self.batches) >= self._max_num_batches:
      print("DEBUG: batch cache: drop: trajs: {} steps: {}".format(self.batches[0].traj_count(), self.batches[0].step_count()))
      pop_batch = self.batches.popleft()
      self._step_ct -= pop_batch.step_count()
      self._traj_ct -= pop_batch.traj_count()
      assert self._step_ct >= 0
      assert self._traj_ct >= 0
      pop_batch.cleanup()
    self._step_ct += new_batch.step_count()
    self._traj_ct += new_batch.traj_count()
    new_batch_idx = len(self.batches)
    self.batches.append(new_batch)
    self.online_batch_idx = new_batch_idx
    assert self.online_batch_idx == len(self.batches) - 1
    print("DEBUG: batch cache: append: num batches: {} total trajs: {} total steps: {}".format(
        len(self.batches), self.traj_count(), self.step_count()))

  def vectorize(self, est_val_adv=False):
    xbatch_sizes = []
    total_xbatch_size = 0
    for batch_idx in range(len(self.batches)):
      batch = self.batches[batch_idx]
      if not batch.is_vectorized():
        batch.vectorize(est_val_adv=est_val_adv)
      assert batch.is_vectorized()
      xbatch_size = batch.step_count() + batch.traj_count()
      xbatch_sizes.append(xbatch_size)
      total_xbatch_size += xbatch_size
    xbatch_offsets = [0] + list(np.cumsum(xbatch_sizes))

    self.xbatch_sizes = xbatch_sizes
    self.xbatch_offsets = xbatch_offsets

  def vectorize_categorical(self):
    #if self.obs_buffer is not None:
    #  del self.obs_buffer
    #  self.obs_buffer = None
    #if self.act_idx_buffer is not None:
    #  del self.act_idx_buffer
    #  self.act_idx_buffer = None

    xbatch_sizes = []
    total_xbatch_size = 0
    for batch_idx in range(len(self.batches)):
      batch = self.batches[batch_idx]
      if not batch.is_vectorized():
        batch.vectorize_categorical()
      xbatch_size = batch.step_count() + batch.traj_count()
      xbatch_sizes.append(xbatch_size)
      total_xbatch_size += xbatch_size
    xbatch_offsets = [0] + list(np.cumsum(xbatch_sizes))

    self.xbatch_sizes = xbatch_sizes
    self.xbatch_offsets = xbatch_offsets

    if False:
      obs_buffer = torch.cuda.ByteTensor(*tuple([total_xbatch_size] + list(self.batches[0].trajs[0].init_obs.shape)))
      act_idx_buffer = torch.LongTensor(total_xbatch_size)

      for batch_idx in range(len(self.batches)):
        batch = self.batches[batch_idx]
        xbatch_size = xbatch_sizes[batch_idx]
        start_index = xbatch_offsets[batch_idx]
        end_index = xbatch_offsets[batch_idx+1]
        obs_buffer[start_index:end_index,:].copy_(batch.obs_buffer)
        act_idx_buffer[start_index:end_index,].copy_(batch.act_idx_buffer)

      torch.cuda.synchronize()

      print("DEBUG: cache obs buffer size: {}".format(obs_buffer.size()))

      self.obs_buffer = obs_buffer
      self.act_idx_buffer = act_idx_buffer

  def reweight_categorical(self, online_policy, clip_weight=1.0, chunk_size=32):
    for batch_idx in range(len(self.batches)):
      if batch_idx == self.online_batch_idx:
        continue
      batch = self.batches[batch_idx]
      batch.reweight_categorical(online_policy, clip_weight=clip_weight, chunk_size=chunk_size)

  def sample_batch(self):
    num_batches = len(self.batches)
    assert num_batches <= self._max_num_batches
    batch_idx = int(np.random.choice(num_batches))
    batch_offset = self.xbatch_offsets[batch_idx]
    return batch_idx, batch_offset, self.batches[batch_idx]

  def sample_minibatch(self, minibatch_size, nsteps=1, discount=1.0, res_scale=None, res_weight=True, categorical=False):
    num_batches = len(self.batches)
    idxs = []
    nstep_idxs = []
    obs_ks = []
    obs_kpns = []
    act_ks = []
    v_rets = []
    q_rets = []
    dones = []
    for _ in range(minibatch_size):
      batch_idx, batch_offset, batch = self.sample_batch()
      if batch_idx == self.online_batch_idx:
        batch_res_weight = False
      else:
        batch_res_weight = res_weight
      idx, nstep_idx, v_nres, q_nres, done = batch.sample_transition(nsteps=nsteps, discount=discount, res_scale=res_scale, res_weight=batch_res_weight, strategy="uniform")
      obs_k = torch.unsqueeze(batch.obs_buffer[idx,:], dim=0)
      obs_kpn = torch.unsqueeze(batch.obs_buffer[nstep_idx,:], dim=0)
      #if batch.act_buffer is not None:
      if categorical:
        act_k = int(batch.act_idx_buffer[idx])
      else:
        act_k = torch.unsqueeze(batch.act_buffer[idx,:], dim=0)
      idxs.append(batch_offset + idx)
      nstep_idxs.append(batch_offset + nstep_idx)
      obs_ks.append(obs_k)
      obs_kpns.append(obs_kpn)
      act_ks.append(act_k)
      v_rets.append(v_nres)
      q_rets.append(q_nres)
      dones.append(done)
    obs_ks = torch.cat(obs_ks, dim=0)
    obs_kpns = torch.cat(obs_kpns, dim=0)
    #if batch.act_buffer is None:
    if categorical:
      act_ks = torch.from_numpy(np.array(act_ks)).type(torch.LongTensor)
    else:
      act_ks = torch.cat(act_ks, dim=0)
    v_rets = torch.from_numpy(np.array(v_rets)).type(torch.FloatTensor)
    q_rets = torch.from_numpy(np.array(q_rets)).type(torch.FloatTensor)
    dones = torch.from_numpy(np.array(dones)).type(torch.FloatTensor)
    return idxs, nstep_idxs, obs_ks, obs_kpns, act_ks, v_rets, q_rets, dones

  def sample_minibatch_new(self, minibatch_size, nsteps=1, discount=0.99, res_scale=1.0, res_weight=True):
    num_batches = len(self.batches)
    minibatch = {}
    idxs = []
    nstep_idxs = []
    obs_ks = []
    obs_kpns = []
    act_ks = []
    val_ks = []
    adv_ks = []
    nadv_ks = []
    v_rets = []
    q_rets = []
    dones = []
    for _ in range(minibatch_size):
      batch_idx, batch_offset, batch = self.sample_batch()
      if batch_idx == self.online_batch_idx:
        batch_res_weight = False
      else:
        batch_res_weight = res_weight
      idx, nstep_idx, v_nres, q_nres, done = batch.sample_transition(nsteps=nsteps, discount=discount, res_scale=res_scale, res_weight=batch_res_weight, strategy="uniform")
      obs_k = torch.unsqueeze(batch.obs_buffer[idx,:], dim=0)
      obs_kpn = torch.unsqueeze(batch.obs_buffer[nstep_idx,:], dim=0)
      if batch.act_buffer is not None:
        act_k = torch.unsqueeze(batch.act_buffer[idx,:], dim=0)
      else:
        act_k = int(batch.act_idx_buffer[idx])
      idxs.append(batch_offset + idx)
      nstep_idxs.append(batch_offset + nstep_idx)
      obs_ks.append(obs_k)
      obs_kpns.append(obs_kpn)
      act_ks.append(act_k)
      val_ks.append(float(batch.val_buffer[idx]))
      adv_ks.append(float(batch.adv_buffer[idx]))
      nadv_ks.append(float(batch.nadv_buffer[idx]))
      v_rets.append(v_nres)
      q_rets.append(q_nres)
      dones.append(done)
    obs_ks = torch.cat(obs_ks, dim=0)
    obs_kpns = torch.cat(obs_kpns, dim=0)
    if batch.act_buffer is not None:
      act_ks = torch.cat(act_ks, dim=0)
    else:
      act_ks = torch.from_numpy(np.array(act_ks)).type(torch.LongTensor)
    val_ks = torch.from_numpy(np.array(val_ks)).type(torch.FloatTensor)
    adv_ks = torch.from_numpy(np.array(adv_ks)).type(torch.FloatTensor)
    nadv_ks = torch.from_numpy(np.array(nadv_ks)).type(torch.FloatTensor)
    v_rets = torch.from_numpy(np.array(v_rets)).type(torch.FloatTensor)
    q_rets = torch.from_numpy(np.array(q_rets)).type(torch.FloatTensor)
    dones = torch.from_numpy(np.array(dones)).type(torch.FloatTensor)
    minibatch["obs_ks"] = obs_ks
    minibatch["act_ks"] = act_ks
    minibatch["val_ks"] = val_ks
    minibatch["adv_ks"] = adv_ks
    minibatch["nadv_ks"] = nadv_ks
    return minibatch

class SerialReplayCache(object):
  def __init__(self, capacity, env):
    self._capacity = capacity
    self._step_idx = 0
    self._step_count = 0
    self._traj_ctr = 0
    self._obs_buf = torch.from_numpy(np.zeros((capacity,) + env.observation_space.shape, dtype=np.float32))
    self._next_obs_buf = torch.from_numpy(np.zeros((capacity,) + env.observation_space.shape, dtype=np.float32))
    self._act_buf = torch.from_numpy(np.zeros((capacity,) + env.action_space.shape, dtype=np.float32))
    self._act_prob_buf = torch.from_numpy(np.zeros(capacity, dtype=np.float32))
    self._res_buf = torch.from_numpy(np.zeros(capacity, dtype=np.float32))
    self._reset_flag = torch.from_numpy(np.zeros(capacity, dtype=np.float32))
    self._done_flag = torch.from_numpy(np.zeros(capacity, dtype=np.float32))
    self._prev_obs = None
    self._prev_reset = False

  def reset(self, env):
    obs = env.reset()

    idx = self._step_idx
    self._reset_buf[idx] = 1.0

    self._prev_obs = obs
    self._prev_reset = True

    self._traj_ctr += 1

  def step(self, env, policy):
    act, act_prob = policy(self._prev_obs)
    next_obs, res, done, _ = env.step(act)

    idx = self._step_idx
    self._obs_buf[idx,:].copy_(torch.from_numpy(self._prev_obs))
    self._next_obs_buf[idx,:].copy_(torch.from_numpy(next_obs))
    self._act_buf[idx,:].copy_(torch.from_numpy(act))
    self._act_prob_buf[idx] = act_prob
    self._res_buf[idx] = res
    if not self._prev_reset:
      self._reset_buf[idx] = 0.0
    self._done_buf[idx] = 1.0 if done else 0.0

    self._prev_obs = next_obs
    self._prev_reset = False

    self._step_idx = (self._step_idx + 1) % self._capacity
    if self._step_count < self._capacity:
      self._step_count += 1

  def sample_one(self, n):
    idx = np.random.choice(self._step_count)
    # TODO

  def sample_minibatch(self, n):
    for _ in range(n):
      self.sample_one()
    # TODO
