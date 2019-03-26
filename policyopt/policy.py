from policyopt.autodiff import *

import gym
import numpy as np
import torch

class UniformBlackboxPolicy(object):
  def __init__(self, action_space):
    self._action_space = action_space

  def __call__(self, obs_buf):
    if isinstance(obs_buf, np.ndarray):
      batch_size = 1
    elif isinstance(obs_buf, torch.Tensor):
      batch_size = obs_buf.size(0)
    else:
      # TODO
      raise NotImplementedError
    acts = []
    act_probs = []
    for _ in range(batch_size):
      act = self._action_space.sample()
      #print("DEBUG: blackbox: act:", type(act), act)
      acts.append(act)
      act_probs.append(None)
    return acts, act_probs

class UniformCategoricalPolicy(object):
  def __init__(self, action_dim):
    self._act_dim = action_dim
    self._act_prob = np.float32(1.0 / float(self._act_dim))

  def __call__(self, obs_buf):
    if isinstance(obs_buf, np.ndarray):
      batch_size = 1
    elif isinstance(obs_buf, torch.Tensor):
      batch_size = obs_buf.size(0)
    else:
      # TODO
      raise NotImplementedError
    act_idxs = []
    act_probs = []
    for _ in range(batch_size):
      act_idx = np.random.choice(self._act_dim)
      act_idxs.append(act_idx)
      act_probs.append(self._act_prob)
    return act_idxs, act_probs

class BatchTorchPolicy(object):
  def __init__(self, build_action):
    self._build_action = build_action

  def __call__(self, obs_buf):
    if isinstance(obs_buf, np.ndarray):
      obs_buf = torch.unsqueeze(torch.from_numpy(obs_buf), 0)
    elif isinstance(obs_buf, torch.Tensor):
      pass
    else:
      # TODO
      raise NotImplementedError
    obs = const_var(obs_buf)
    action, action_prob = self._build_action(obs)
    return action, action_prob

  def serialize_flat_param(self, dst_param):
    # TODO
    raise NotImplementedError

  def deserialize_flat_param(self, src_param):
    # TODO
    raise NotImplementedError

class DiagonalGaussianPolicy(BatchTorchPolicy):
  def __init__(self, build_mean_variance):
    import scipy.stats
    def build_action(obs_input):
      mean_var, variance_var = build_mean_variance(obs_input)
      mean, variance = mean_var.data.cpu().numpy(), variance_var.data.cpu().numpy()
      covariance = np.diag(variance)
      z = np.random.multivariate_normal(mean, covariance)
      prob = scipy.stats.multivariate_normal.pdf(z, mean, covariance)
      return z, prob
    super().__init__(build_action)

class GeneratorPolicy(BatchTorchPolicy):
  def __init__(self, generator_fn, density_fn):
    def build_action(obs):
      act_var = generator_fn(obs)
      act_buf = act_var.data.cpu().numpy()
      if density_fn is not None:
        prob_var = density_fn(obs)
        prob_buf = prob_var.data.cpu().numpy()
      else:
        prob_buf = None
      batch_size = act_buf.shape[0]
      acts = []
      probs = []
      for i in range(batch_size):
        acts.append(act_buf[i,:])
        probs.append(None)
      return acts, probs
    super().__init__(build_action)

class GeneratorModePolicy(BatchTorchPolicy):
  def __init__(self, generator_fn, log_density_fn, step_size=None, num_iters=None):
    def policy_fn(obs_var):
      act_var = generator_fn(obs_var)
      for _ in range(num_iters):
        log_prob_var = log_density_fn(obs_var, act_var)
        act_grad = adjoint(log_prob_var, [act_var])[0]
        act_var += step_size * act_grad
      act = act_var.data.cpu().numpy()
      act_prob = np.exp(act)
      return act, act_prob
    super().__init__(policy_fn)

if __name__ == "__main__":
  policy = UniformCategoricalPolicy(8)
  act_idx, act_prob = policy(None)
  policy = DiagonalGaussianPolicy(None)
