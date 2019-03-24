# policyopt_torch

[![guppybot.org](https://guppybot.org/ci/peterhj/arm-pytorch/badge.svg)](https://guppybot.org/ci/peterhj/arm-pytorch)

This contains an implementation of ARM for discrete action space envs
(https://arxiv.org/abs/1710.11424v2).

## Requirements

Tested with python 3.5.

- gym commit [0c91364cd4a7ea70f242a28b85c3aea2d74aa35a](https://github.com/openai/gym/tree/0c91364cd4a7ea70f242a28b85c3aea2d74aa35a)
- numpy 1.13, 1.14 or newer
- opencv-python
- pytorch 0.3.1

## Atari

Run `python ./train_atari_arm.py` with the gym Atari envs installed.
See the comments in `train_atari_arm.py` for the various options.

## Doom

Similarly, run `python ./train_doom_arm.py`. ViZDoom experiments use
slightly customized versions of doom-py and the envs by @ppaquette:

    https://github.com/peterhj/doom-py/tree/peterhj-depth
    https://github.com/peterhj/gym-doom/tree/peterhj-rllab
