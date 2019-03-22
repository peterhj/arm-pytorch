#!/usr/bin/env python3.5

import sys
sys.path.append("./")

import policyopt
from policyopt.autodiff import *
from policyopt.discrete_arm import DiscreteARM
from policyopt.functional import build_atari_84x84_fn
from policyopt.wrapper import wrap_atari_84x84_v2
from policyopt.wrapper_atari import wrap_atari_dqn

import gym
import torch

def main():
  env_id = "PongNoFrameskip-v4"

  # This option controls the number of recent frames to stack in an
  # observation.
  #HISTORY_LEN = 1
  HISTORY_LEN = 4
  print("DEBUG: config: history len:", HISTORY_LEN)

  # This option controls the Pong occlusion setup from the paper.
  PREPROC_PONG_MASK = False
  #PREPROC_PONG_MASK = True
  print("DEBUG: config: preproc pong mask:", PREPROC_PONG_MASK)

  # Unused (remains for backward compatibility).
  PREPROC_BREAKOUT_MASK = False
  #PREPROC_BREAKOUT_MASK = True
  print("DEBUG: config: preproc breakout mask:", PREPROC_BREAKOUT_MASK)

  # Unused (remains for backward compatibility).
  FRAME_FLICKER = False
  #FRAME_FLICKER = True
  print("DEBUG: config: frame flicker:", FRAME_FLICKER)

  # Unused.
  #FRAME_REPEAT = False
  ##FRAME_REPEAT = True
  #print("DEBUG: config: frame repeat:", FRAME_REPEAT)

  env = gym.make(env_id)
  h = env.seed()
  print("DEBUG: config: env seed:", h)
  env = wrap_atari_84x84_v2(
      env,
      preproc_pong_mask=PREPROC_PONG_MASK,
      preproc_breakout_mask=PREPROC_BREAKOUT_MASK,
      frame_flicker=FRAME_FLICKER,
      history_len=HISTORY_LEN)

  input_chan = HISTORY_LEN
  act_dim = env.action_space.n

  arm_cfg = {
    # If more than 1 cached batch, then off-policy replay is on.
    "num_cached_batches": 1,
    # Should be kept as "uniform" (prioritized sampling is not implemented).
    "sampling_strategy": "uniform",
    # Number of steps in n-step estimator.
    "nsteps": 1,
    # Discount factor of returns.
    "discount": 0.99,
    # Reward scailng; None is the default unit scaling.
    "res_scale": None,
    # Number of Adam iterations to run on the very first ARM batch
    # (corresponds to the uniform policy).
    "initial_num_arm_iters": 3000,
    # Number of Adam iterations to run on the later ARM batches.
    "num_arm_iters": 3000,
    # Adam minibatch size.
    "minibatch_size": 32,
    # Target value function parameters are updated via moving average with
    # this rate.
    "arm_target_step_size": 0.01,
  }
  print("DEBUG: arm cfg:", arm_cfg)
  # Batch configuration, main setting is the number of transitions per batch.
  batch_cfg = {
    "sample_size": 12500,
    "num_trajs": None,
  }
  print("DEBUG: batch cfg:", batch_cfg)
  # Unused (remains for backward compatibility).
  eval_cfg = {
    "sample_size": None,
    "num_trajs": 30,
  }
  # Adam optimizer configuration.
  opt_cfg = {
    "batch_size": 32,
    "minibatch_size": 32,
    "step_size": 1.0e-4,
    "decay_rate_1": 0.1,
    "decay_rate_2": 0.001,
    "epsilon": 1.0e-8,
  }
  print("DEBUG: opt cfg:", opt_cfg)

  prev_v_param_vars, prev_v_init_fns, prev_v_fn = build_atari_84x84_fn(input_chan, 1)
  initialize_vars(prev_v_param_vars, prev_v_init_fns)
  prev_ccq_param_vars, prev_ccq_init_fns, prev_ccq_fn = build_atari_84x84_fn(input_chan, act_dim)
  initialize_vars(prev_ccq_param_vars, prev_ccq_init_fns)
  tg_v_param_vars, tg_v_init_fns, tg_v_fn = build_atari_84x84_fn(input_chan, 1)
  initialize_vars(tg_v_param_vars, tg_v_init_fns)
  v_param_vars, v_init_fns, v_fn = build_atari_84x84_fn(input_chan, 1)
  initialize_vars(v_param_vars, v_init_fns)
  ccq_param_vars, ccq_init_fns, ccq_fn = build_atari_84x84_fn(input_chan, act_dim)
  initialize_vars(ccq_param_vars, ccq_init_fns)

  arm = DiscreteARM(arm_cfg, batch_cfg, eval_cfg, opt_cfg)
  arm.reset(env, prev_v_param_vars, prev_v_fn, prev_ccq_param_vars, prev_ccq_fn, tg_v_param_vars, tg_v_fn, v_param_vars, v_fn, ccq_param_vars, ccq_fn)

  total_step_limit = 10000000
  #total_step_limit = 20000000
  total_step_count = 0
  while total_step_count <= total_step_limit:
    total_step_count += arm.step(env)

  env.close()

if __name__ == "__main__":
  main()
