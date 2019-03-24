#!/usr/bin/env python3.5

import sys
sys.path.append("./")

import policyopt
from policyopt.autodiff import *
from policyopt.discrete_arm import DiscreteARM
from policyopt.functional import build_doom_84x84_fn, build_doom_44x44_fn, build_doom_20x20_fn, build_doom_1x1_fn
from policyopt.utils import *
from policyopt.wrapper_doom import wrap_doom_84x84, wrap_doom_84x84_v2, wrap_doom_44x44_v2, wrap_doom_20x20_v2, wrap_doom_1x1_v2

import gym
import ppaquette_gym_doom
import torch

torch.set_num_threads(1)
#torch.backends.cudnn.benchmark = True

def main():
  gseed = None
  if len(sys.argv) > 1:
    gseed = int(sys.argv[1])
    set_global_seeds(gseed)
  print("DEBUG: global seed:", gseed)

  #env_id = "ppaquette/DoomHealthGathering-v0"
  #env_id = "ppaquette/DoomMyWayHome-v0"
  env_id = "ppaquette/DoomCorridor-v0"

  IMAGE_DIM = 84
  print("DEBUG: config: image dim:", IMAGE_DIM)

  FRAME_HISTORY = 4
  #FRAME_HISTORY = 1
  print("DEBUG: config: frame history:", FRAME_HISTORY)

  PREPROC_MASK = False
  #PREPROC_MASK = True
  print("DEBUG: config: preproc mask:", PREPROC_MASK)

  PREPROC_MASK_BIG = False
  #PREPROC_MASK_BIG = True
  print("DEBUG: config: preproc mask big:", PREPROC_MASK_BIG)

  PREPROC_MASK_OUT = False
  #PREPROC_MASK_OUT = True
  print("DEBUG: config: preproc mask out:", PREPROC_MASK_OUT)

  ACTION_DELAY = 0
  #ACTION_DELAY = 3
  #ACTION_DELAY = 4
  #ACTION_DELAY = 10
  print("DEBUG: config: action delay:", ACTION_DELAY)

  ZCLIP_FEAT = False
  print("DEBUG: config: zclip feat:", ZCLIP_FEAT)

  DEPTH_FEAT = False
  #DEPTH_FEAT = True
  print("DEBUG: config: depth feat:", DEPTH_FEAT)

  FRAME_FLICKER = False
  #FRAME_FLICKER = True
  print("DEBUG: config: frame flicker:", FRAME_FLICKER)

  test_env = gym.make(env_id)
  test_env.close()

  env = gym.make(env_id)
  h = env.seed()
  print("DEBUG: env seed:", h)
  if IMAGE_DIM == 84:
    env = wrap_doom_84x84_v2(
        env,
        frame_skip=4,
        action_delay=ACTION_DELAY,
        preproc_mask=PREPROC_MASK,
        preproc_mask_big=PREPROC_MASK_BIG,
        preproc_mask_out=PREPROC_MASK_OUT,
        depth=DEPTH_FEAT,
        frame_flicker=FRAME_FLICKER,
        history_len=FRAME_HISTORY)
  else:
    raise NotImplementedError

  input_chan = FRAME_HISTORY * 3
  #input_chan = FRAME_HISTORY * 1   # for depth features.
  act_dim = env.action_space.n
  print("DEBUG: act dim:", act_dim)

  arm_cfg = {
    "num_cached_batches": 1,
    "sampling_strategy": "uniform",
    "nsteps": 5,
    "discount": 0.99,
    "res_scale": 0.01,
    "initial_num_arm_iters": 3000,
    "num_arm_iters": 3000,
    "minibatch_size": 32,
    "arm_target_step_size": 0.01,
  }
  print("DEBUG: arm cfg: {}".format(arm_cfg))
  batch_cfg = {
    "sample_size": 12500,
    "num_trajs": None,
  }
  print("DEBUG: batch cfg: {}".format(batch_cfg))
  eval_cfg = {
    "sample_size": None,
    "num_trajs": 30,
  }
  opt_cfg = {
    "batch_size": 32,
    "minibatch_size": 32,
    "step_size": 1.0e-5,
    "decay_rate_1": 0.1,
    "decay_rate_2": 0.001,
    "epsilon": 1.0e-8,
  }
  print("DEBUG: adam cfg: {}".format(opt_cfg))

  if IMAGE_DIM == 84:
    prev_v_param_vars, prev_v_init_fns, prev_v_fn = build_doom_84x84_fn(input_chan, 1)
    prev_ccq_param_vars, prev_ccq_init_fns, prev_ccq_fn = build_doom_84x84_fn(input_chan, act_dim)
    tg_v_param_vars, tg_v_init_fns, tg_v_fn = build_doom_84x84_fn(input_chan, 1)
    v_param_vars, v_init_fns, v_fn = build_doom_84x84_fn(input_chan, 1)
    ccq_param_vars, ccq_init_fns, ccq_fn = build_doom_84x84_fn(input_chan, act_dim)
  elif IMAGE_DIM == 44:
    prev_v_param_vars, prev_v_init_fns, prev_v_fn = build_doom_44x44_fn(input_chan, 1)
    prev_ccq_param_vars, prev_ccq_init_fns, prev_ccq_fn = build_doom_44x44_fn(input_chan, act_dim)
    tg_v_param_vars, tg_v_init_fns, tg_v_fn = build_doom_44x44_fn(input_chan, 1)
    v_param_vars, v_init_fns, v_fn = build_doom_44x44_fn(input_chan, 1)
    ccq_param_vars, ccq_init_fns, ccq_fn = build_doom_44x44_fn(input_chan, act_dim)
  elif IMAGE_DIM == 20:
    prev_v_param_vars, prev_v_init_fns, prev_v_fn = build_doom_20x20_fn(input_chan, 1)
    prev_ccq_param_vars, prev_ccq_init_fns, prev_ccq_fn = build_doom_20x20_fn(input_chan, act_dim)
    tg_v_param_vars, tg_v_init_fns, tg_v_fn = build_doom_20x20_fn(input_chan, 1)
    v_param_vars, v_init_fns, v_fn = build_doom_20x20_fn(input_chan, 1)
    ccq_param_vars, ccq_init_fns, ccq_fn = build_doom_20x20_fn(input_chan, act_dim)
  elif IMAGE_DIM == 1:
    prev_v_param_vars, prev_v_init_fns, prev_v_fn = build_doom_1x1_fn(input_chan, 1)
    prev_ccq_param_vars, prev_ccq_init_fns, prev_ccq_fn = build_doom_1x1_fn(input_chan, act_dim)
    tg_v_param_vars, tg_v_init_fns, tg_v_fn = build_doom_1x1_fn(input_chan, 1)
    v_param_vars, v_init_fns, v_fn = build_doom_1x1_fn(input_chan, 1)
    ccq_param_vars, ccq_init_fns, ccq_fn = build_doom_1x1_fn(input_chan, act_dim)
  else:
    raise NotImplementedError
  initialize_vars(prev_v_param_vars, prev_v_init_fns)
  initialize_vars(prev_ccq_param_vars, prev_ccq_init_fns)
  initialize_vars(tg_v_param_vars, tg_v_init_fns)
  initialize_vars(v_param_vars, v_init_fns)
  initialize_vars(ccq_param_vars, ccq_init_fns)

  arm = DiscreteARM(arm_cfg, batch_cfg, eval_cfg, opt_cfg)
  arm.reset(env, prev_v_param_vars, prev_v_fn, prev_ccq_param_vars, prev_ccq_fn, tg_v_param_vars, tg_v_fn, v_param_vars, v_fn, ccq_param_vars, ccq_fn)

  total_step_limit = 2000000
  #total_step_limit = 4000000
  #total_step_limit = 6000000
  #total_step_limit = 10000000
  total_step_count = 0
  while total_step_count <= total_step_limit:
    total_step_count += arm.step(env)
  print("DEBUG: training finished")
  sys.stdout.flush()

  env.close()
  print("DEBUG: closed env")
  sys.stdout.flush()

if __name__ == "__main__":
  main()
