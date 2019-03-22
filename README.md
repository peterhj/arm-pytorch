# policyopt_torch

[![guppybot.org](https://guppybot.org/ci/peterhj/arm-pytorch/badge.svg)](https://guppybot.org/ci/peterhj/arm-pytorch)

This contains an implementation of ARM for discrete action space envs.

## Requirements

Tested on python 3.5, not sure about 2.x compat.

- gym commit [0c91364cd4a7ea70f242a28b85c3aea2d74aa35a](https://github.com/openai/gym/tree/0c91364cd4a7ea70f242a28b85c3aea2d74aa35a)
- numpy 1.13, 1.14 or newer
- opencv-python
- pytorch 0.3.1

## Atari

Run `python ./train_atari_arm.py` with the gym Atari envs installed.
See the comments in `train_atari_arm.py` for the various options.
