#!/usr/bin/env python3.5

from policyopt.autodiff import *

import numpy as np
import torch
import torch.cuda
import torch.nn.functional

# TODO

class ParamsBuilder(object):
  def __init__(self):
    self._keys = []
    self._kvs = {}

  def get_var(self, key, init_fn):
    if key not in self._kvs:
      self._keys.append(key)
      self._kvs[key] = (var(init_fn()), init_fn)
    v, _ = self._kvs[key]
    return v

  def build(self):
    param_vars = []
    param_init_fns = []
    for key in reversed(self._keys):
      v, init_fn = self._kvs[key]
      param_vars.append(v)
      param_init_fns.append(init_fn)
    return param_vars, param_init_fns

def build_zeros_init(dim, dtype):
  def init_fn():
    return torch.zeros(*dim).type(dtype)
  return init_fn

def build_normc_conv2d_init(kernel_dim, dtype):
  out_chan, in_chan, kernel_h, kernel_w = kernel_dim
  fan_in = in_chan * kernel_h * kernel_w
  fan_out = out_chan
  def init_fn():
    x = np.random.randn(fan_out, fan_in).astype(np.float32)
    x /= np.sqrt(np.square(x).sum(axis=1, keepdims=True))
    x = np.reshape(x, kernel_dim)
    return torch.from_numpy(x).type(dtype)
  return init_fn

def build_xavier_conv2d_init(kernel_dim, dtype):
  out_chan, in_chan, kernel_h, kernel_w = kernel_dim
  fan_in = in_chan * kernel_h * kernel_w
  fan_out = out_chan
  def init_fn():
    half_width = np.sqrt(6.0 / float(fan_in + fan_out))
    x = np.random.uniform(low=-half_width, high=half_width, size=kernel_dim).astype(np.float32)
    return torch.from_numpy(x).type(dtype)
  return init_fn

def build_normal_linear_init(kernel_dim, mean, std, dtype):
  def init_fn():
    x = np.random.normal(loc=mean, scale=std, size=kernel_dim).astype(np.float32)
    return torch.from_numpy(x).type(dtype)
  return init_fn

def build_normc_linear_init(kernel_dim, dtype=None, std=1.0):
  def init_fn():
    x = np.random.randn(*kernel_dim).astype(np.float32)
    x *= std / np.sqrt(np.square(x).sum(axis=1, keepdims=True))
    return torch.from_numpy(x).type(dtype)
  return init_fn

def build_xavier_linear_init(kernel_dim, dtype):
  out_chan, in_chan = kernel_dim
  def init_fn():
    half_width = np.sqrt(6.0 / float(in_chan + out_chan))
    x = np.random.uniform(low=-half_width, high=half_width, size=kernel_dim).astype(np.float32)
    return torch.from_numpy(x).type(dtype)
  return init_fn

def compose(f, g):
  return lambda x: g(f(x))

def conv2d(f, w, **kwargs):
  #return compose(f, lambda x: torch.nn.functional.conv2d(x, w, **kwargs))
  def conv2d_fn(x):
    #print("DEBUG: conv2d:", w.size(), x.size())
    y = torch.nn.functional.conv2d(x, w, **kwargs)
    #print("DEBUG:   ", y.size())
    return y
  return compose(f, conv2d_fn)

def linear(f, w, **kwargs):
  #return compose(f, lambda x: torch.nn.functional.linear(x, w, **kwargs))
  def linear_fn(x):
    y = torch.nn.functional.linear(x, w, **kwargs)
    #print("DEBUG: linear:", w.size(), x.size(), y.size())
    return y
  return compose(f, linear_fn)

def relu(f):
  return compose(f, lambda x: torch.nn.functional.relu(x))

def tanh(f):
  return compose(f, lambda x: torch.nn.functional.tanh(x))

def scale(f, scale_factor):
  return compose(f, lambda x: scale_factor * x)

def flat(f):
  #return compose(f, lambda x: x.view(x.size(0), -1))
  def flat_fn(x):
    y = x.view(x.size(0), -1)
    #print("DEBUG: flat:", x.size(), y.size())
    return y
  return compose(f, flat_fn)

def squeeze(f, dim):
  def squeeze_fn(x):
    y = torch.squeeze(x, dim=dim)
    return y
  return compose(f, squeeze_fn)

def concat2(f, dim):
  def concat_fn(x1, x2):
    return torch.cat((x1, x2), dim=dim)
  return lambda x1, x2: f(concat_fn(x1, x2))

def glue2(f, g1, g2, glue):
  def glue_fn(x):
    x1 = g1(x)
    x2 = g2(x)
    y = glue(x1, x2)
    return y
  return compose(f, glue_fn)

def glue3(f, g1, g2, g3, glue):
  def glue_fn(x):
    x1 = g1(x)
    x2 = g2(x)
    x3 = g3(x)
    y = glue(x1, x2, x3)
    return y
  return compose(f, glue_fn)

def build_atari_84x84_fn(input_dim, output_dim, dtype=torch.cuda.FloatTensor):
  params = ParamsBuilder()
  f = lambda obs: obs.type(dtype)
  f = scale(f, 1.0 / 255.0)
  w1 = params.get_var("w1", build_xavier_conv2d_init((16, input_dim, 8, 8), dtype))
  b1 = params.get_var("b1", lambda: torch.zeros(16).type(dtype))
  f = conv2d(f, w1, bias=b1, stride=4, padding=0)
  f = relu(f)
  w2 = params.get_var("w2", build_xavier_conv2d_init((32, 16, 4, 4), dtype))
  b2 = params.get_var("b2", lambda: torch.zeros(32).type(dtype))
  f = conv2d(f, w2, bias=b2, stride=2, padding=0)
  f = relu(f)
  f = flat(f)
  a4 = params.get_var("a4", build_xavier_linear_init((256, 32 * 9 * 9), dtype))
  b4 = params.get_var("b4", lambda: torch.zeros(256).type(dtype))
  f = linear(f, a4, bias=b4)
  f = relu(f)
  a5 = params.get_var("a5", build_xavier_linear_init((output_dim, 256), dtype))
  b5 = params.get_var("b5", lambda: torch.zeros(output_dim).type(dtype))
  f = linear(f, a5, bias=b5)
  param_vars, param_init_fns = params.build()
  return param_vars, param_init_fns, f

def build_doom_84x84_fn(input_dim, output_dim, dtype=torch.cuda.FloatTensor):
  params = ParamsBuilder()
  #def obs_fn(x):
  #  #print("DEBUG: obs:", x.size())
  #  return x.type(dtype)
  #f = obs_fn
  f = lambda obs: obs.type(dtype)
  f = scale(f, 1.0 / 255.0)
  w1 = params.get_var("w1", build_xavier_conv2d_init((32, input_dim, 8, 8), dtype))
  b1 = params.get_var("b1", lambda: torch.zeros(32).type(dtype))
  f = conv2d(f, w1, bias=b1, stride=4, padding=0)
  f = relu(f)
  w2 = params.get_var("w2", build_xavier_conv2d_init((32, 32, 4, 4), dtype))
  b2 = params.get_var("b2", lambda: torch.zeros(32).type(dtype))
  f = conv2d(f, w2, bias=b2, stride=2, padding=0)
  f = relu(f)
  w3 = params.get_var("w3", build_xavier_conv2d_init((32, 32, 3, 3), dtype))
  b3 = params.get_var("b3", lambda: torch.zeros(32).type(dtype))
  f = conv2d(f, w3, bias=b3, stride=1, padding=1)
  f = relu(f)
  f = flat(f)
  a4 = params.get_var("a4", build_xavier_linear_init((1024, 32 * 9 * 9), dtype))
  b4 = params.get_var("b4", lambda: torch.zeros(1024).type(dtype))
  f = linear(f, a4, bias=b4)
  f = relu(f)
  a5 = params.get_var("a5", build_xavier_linear_init((output_dim, 1024), dtype))
  b5 = params.get_var("b5", lambda: torch.zeros(output_dim).type(dtype))
  f = linear(f, a5, bias=b5)
  param_vars, param_init_fns = params.build()
  return param_vars, param_init_fns, f

def build_doom_44x44_fn(input_dim, output_dim, dtype=torch.cuda.FloatTensor):
  params = ParamsBuilder()
  #def obs_fn(x):
  #  #print("DEBUG: obs:", x.size())
  #  return x.type(dtype)
  #f = obs_fn
  f = lambda obs: obs.type(dtype)
  f = scale(f, 1.0 / 255.0)
  w1 = params.get_var("w1", build_xavier_conv2d_init((32, input_dim, 8, 8), dtype))
  b1 = params.get_var("b1", lambda: torch.zeros(32).type(dtype))
  f = conv2d(f, w1, bias=b1, stride=4, padding=0)
  f = relu(f)
  w2 = params.get_var("w2", build_xavier_conv2d_init((32, 32, 4, 4), dtype))
  b2 = params.get_var("b2", lambda: torch.zeros(32).type(dtype))
  f = conv2d(f, w2, bias=b2, stride=2, padding=0)
  f = relu(f)
  w3 = params.get_var("w3", build_xavier_conv2d_init((32, 32, 3, 3), dtype))
  b3 = params.get_var("b3", lambda: torch.zeros(32).type(dtype))
  f = conv2d(f, w3, bias=b3, stride=1, padding=1)
  f = relu(f)
  f = flat(f)
  a4 = params.get_var("a4", build_xavier_linear_init((1024, 32 * 4 * 4), dtype))
  b4 = params.get_var("b4", lambda: torch.zeros(1024).type(dtype))
  f = linear(f, a4, bias=b4)
  f = relu(f)
  a5 = params.get_var("a5", build_xavier_linear_init((output_dim, 1024), dtype))
  b5 = params.get_var("b5", lambda: torch.zeros(output_dim).type(dtype))
  f = linear(f, a5, bias=b5)
  param_vars, param_init_fns = params.build()
  return param_vars, param_init_fns, f

def build_doom_28x28_fn(input_dim, output_dim, dtype=torch.cuda.FloatTensor):
  params = ParamsBuilder()
  #def obs_fn(x):
  #  #print("DEBUG: obs:", x.size())
  #  return x.type(dtype)
  #f = obs_fn
  f = lambda obs: obs.type(dtype)
  f = scale(f, 1.0 / 255.0)
  w1 = params.get_var("w1", build_xavier_conv2d_init((32, input_dim, 8, 8), dtype))
  b1 = params.get_var("b1", lambda: torch.zeros(32).type(dtype))
  f = conv2d(f, w1, bias=b1, stride=4, padding=0)
  f = relu(f)
  w2 = params.get_var("w2", build_xavier_conv2d_init((32, 32, 4, 4), dtype))
  b2 = params.get_var("b2", lambda: torch.zeros(32).type(dtype))
  f = conv2d(f, w2, bias=b2, stride=2, padding=0)
  f = relu(f)
  #w3 = params.get_var("w3", build_xavier_conv2d_init((32, 32, 3, 3), dtype))
  #b3 = params.get_var("b3", lambda: torch.zeros(32).type(dtype))
  #f = conv2d(f, w3, bias=b3, stride=1, padding=1)
  #f = relu(f)
  f = flat(f)
  a4 = params.get_var("a4", build_xavier_linear_init((1024, 32 * 2 * 2), dtype))
  b4 = params.get_var("b4", lambda: torch.zeros(1024).type(dtype))
  f = linear(f, a4, bias=b4)
  f = relu(f)
  a5 = params.get_var("a5", build_xavier_linear_init((output_dim, 1024), dtype))
  b5 = params.get_var("b5", lambda: torch.zeros(output_dim).type(dtype))
  f = linear(f, a5, bias=b5)
  param_vars, param_init_fns = params.build()
  return param_vars, param_init_fns, f

def build_doom_20x20_fn(input_dim, output_dim, dtype=torch.cuda.FloatTensor):
  params = ParamsBuilder()
  #def obs_fn(x):
  #  #print("DEBUG: obs:", x.size())
  #  return x.type(dtype)
  #f = obs_fn
  f = lambda obs: obs.type(dtype)
  f = scale(f, 1.0 / 255.0)
  w1 = params.get_var("w1", build_xavier_conv2d_init((32, input_dim, 8, 8), dtype))
  b1 = params.get_var("b1", lambda: torch.zeros(32).type(dtype))
  f = conv2d(f, w1, bias=b1, stride=4, padding=0)
  f = relu(f)
  w2 = params.get_var("w2", build_xavier_conv2d_init((32, 32, 4, 4), dtype))
  b2 = params.get_var("b2", lambda: torch.zeros(32).type(dtype))
  f = conv2d(f, w2, bias=b2, stride=2, padding=0)
  f = relu(f)
  #w3 = params.get_var("w3", build_xavier_conv2d_init((32, 32, 3, 3), dtype))
  #b3 = params.get_var("b3", lambda: torch.zeros(32).type(dtype))
  #f = conv2d(f, w3, bias=b3, stride=1, padding=1)
  #f = relu(f)
  f = flat(f)
  a4 = params.get_var("a4", build_xavier_linear_init((1024, 32 * 1 * 1), dtype))
  b4 = params.get_var("b4", lambda: torch.zeros(1024).type(dtype))
  f = linear(f, a4, bias=b4)
  f = relu(f)
  a5 = params.get_var("a5", build_xavier_linear_init((output_dim, 1024), dtype))
  b5 = params.get_var("b5", lambda: torch.zeros(output_dim).type(dtype))
  f = linear(f, a5, bias=b5)
  param_vars, param_init_fns = params.build()
  return param_vars, param_init_fns, f

def build_doom_1x1_fn(input_dim, output_dim, dtype=torch.cuda.FloatTensor):
  params = ParamsBuilder()
  #def obs_fn(x):
  #  #print("DEBUG: obs:", x.size())
  #  return x.type(dtype)
  #f = obs_fn
  f = lambda obs: obs.type(dtype)
  f = scale(f, 1.0 / 255.0)
  f = flat(f)
  a2 = params.get_var("a2", build_xavier_linear_init((32, input_dim * 1 * 1), dtype))
  b2 = params.get_var("b2", lambda: torch.zeros(32).type(dtype))
  f = linear(f, a2, bias=b2)
  f = relu(f)
  a3 = params.get_var("a3", build_xavier_linear_init((32, 32), dtype))
  b3 = params.get_var("b3", lambda: torch.zeros(32).type(dtype))
  f = linear(f, a3, bias=b3)
  f = relu(f)
  a4 = params.get_var("a4", build_xavier_linear_init((1024, 32 * 1 * 1), dtype))
  b4 = params.get_var("b4", lambda: torch.zeros(1024).type(dtype))
  f = linear(f, a4, bias=b4)
  f = relu(f)
  a5 = params.get_var("a5", build_xavier_linear_init((output_dim, 1024), dtype))
  b5 = params.get_var("b5", lambda: torch.zeros(output_dim).type(dtype))
  f = linear(f, a5, bias=b5)
  param_vars, param_init_fns = params.build()
  return param_vars, param_init_fns, f

if False:
  def build_tanhnet_obs_fn(obs_dim, out_dim, hidden_dims, dtype=torch.cuda.FloatTensor):
    params = ParamsBuilder()
    f = lambda obs: obs.type(dtype)
    # TODO
    raise NotImplementedError
    param_vars, param_init_fns = params.build()
    return param_vars, param_init_fns, f

  def build_tanhnet_obs_act_fn(obs_dim, act_dim, out_dim, hidden_dims, dtype=torch.cuda.FloatTensor):
    params = ParamsBuilder()
    def obs_act_concat_fn(obs, act):
      obs = obs.type(dtype)
      act = act.type(dtype)
      x = torch.cat((obs, act), 1)
      return x
    f = lambda x: x
    # TODO
    raise NotImplementedError
    param_vars, param_init_fns = params.build()
    return param_vars, param_init_fns, lambda obs, act: f(obs_act_concat_fn(obs, act))

if __name__ == "__main__":
  v_param_vars, v_param_init_fns, v_fn = build_doom_84x84_fn(12, 1)
  #print(v_param_vars)
  print(len(v_param_vars), len(v_param_init_fns), v_param_init_fns)
  q_param_vars, q_param_init_fns, q_fn = build_doom_84x84_fn(12, 8)
