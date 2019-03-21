#!/usr/bin/env python3.5

from policyopt.wrapper import *

import cv2
import gym
import numpy as np

from collections import deque

class ProcessFrame84x84Minecraft(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84x84Minecraft, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3))

    def _observation(self, obs):
        return ProcessFrame84x84Minecraft.process(obs)

    @staticmethod
    def process(frame):
        #assert frame.shape == (240, 320, 3)
        img = frame.astype(np.float32)
        x_t = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        assert x_t.shape == (84, 84, 3)
        return x_t.astype(np.uint8)

def wrap_minecraft_84x84(env, frame_skip=1, history_len=1):
  if frame_skip > 1:
    env = SkipFrameEnv(env, skip=frame_skip)
  env = ProcessFrame84x84Minecraft(env)
  env = Transpose3DEnv(env)
  if history_len > 1:
    env = StackFrameEnv(env, history_len)
  return env

class ProcessMinecraftRGB20(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessMinecraftRGB20, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(20, 20, 3))

    def _observation(self, obs):
        return ProcessMinecraftRGB20.process(obs)

    @staticmethod
    def process(frame):
        img = frame.astype(np.float32)
        if frame.size == 240 * 320 * 3:
            img = np.reshape(frame, (240, 320, 3)).astype(np.float32)
            img_color = img
        else:
            assert False, "Unknown resolution: {}".format(frame.size)
        x_t = cv2.resize(img_color, (20, 20), interpolation=cv2.INTER_AREA)
        assert x_t.shape == (20, 20, 3)
        return x_t.astype(np.uint8)

def wrap_minecraft_20x20_v2(env, frame_skip=1, history_len=1, zclip=False):
  if frame_skip > 1:
    env = SkipFrameEnv(env, skip=frame_skip)
  if zclip:
    raise NotImplementedError
  else:
    env = ProcessMinecraftRGB20(env)
  env = Transpose3DEnv(env)
  if history_len > 1:
    env = StackFrameEnv(env, history_len)
  return env

class ProcessMinecraftRGB28(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessMinecraftRGB28, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(28, 28, 3))

    def _observation(self, obs):
        return ProcessMinecraftRGB28.process(obs)

    @staticmethod
    def process(frame):
        img = frame.astype(np.float32)
        if frame.size == 240 * 320 * 3:
            img = np.reshape(frame, (240, 320, 3)).astype(np.float32)
            img_color = img
        else:
            assert False, "Unknown resolution: {}".format(frame.size)
        x_t = cv2.resize(img_color, (28, 28), interpolation=cv2.INTER_AREA)
        assert x_t.shape == (28, 28, 3)
        return x_t.astype(np.uint8)

def wrap_minecraft_28x28_v2(env, frame_skip=1, history_len=1, zclip=False):
  if frame_skip > 1:
    env = SkipFrameEnv(env, skip=frame_skip)
  if zclip:
    raise NotImplementedError
  else:
    env = ProcessMinecraftRGB28(env)
  env = Transpose3DEnv(env)
  if history_len > 1:
    env = StackFrameEnv(env, history_len)
  return env

class ProcessMinecraftRGB44(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessMinecraftRGB44, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(44, 44, 3))

    def _observation(self, obs):
        return ProcessMinecraftRGB44.process(obs)

    @staticmethod
    def process(frame):
        img = frame.astype(np.float32)
        if frame.size == 240 * 320 * 3:
            img = np.reshape(frame, (240, 320, 3)).astype(np.float32)
            img_color = img
        else:
            assert False, "Unknown resolution: {}".format(frame.size)
        x_t = cv2.resize(img_color, (44, 44), interpolation=cv2.INTER_AREA)
        assert x_t.shape == (44, 44, 3)
        return x_t.astype(np.uint8)

def wrap_minecraft_44x44_v2(env, frame_skip=1, history_len=1, zclip=False):
  if frame_skip > 1:
    env = SkipFrameEnv(env, skip=frame_skip)
  if zclip:
    raise NotImplementedError
  else:
    env = ProcessMinecraftRGB44(env)
  env = Transpose3DEnv(env)
  if history_len > 1:
    env = StackFrameEnv(env, history_len)
  return env

class ProcessMinecraftRGBD(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessMinecraftRGBD, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3))

    def _observation(self, obs):
        return ProcessMinecraftRGBD.process(obs)

    @staticmethod
    def process(frame):
        img = frame.astype(np.float32)
        if frame.size == 240 * 320 * 4:
            img = np.reshape(frame, (240, 320, 4)).astype(np.float32)
            img_color = img[:,:,:3]
        elif frame.size == 240 * 320 * 3:
            img = np.reshape(frame, (240, 320, 3)).astype(np.float32)
            img_color = img
        else:
            assert False, "Unknown resolution: {}".format(frame.size)
        x_t = cv2.resize(img_color, (84, 84), interpolation=cv2.INTER_AREA)
        assert x_t.shape == (84, 84, 3)
        return x_t.astype(np.uint8)

class ProcessMinecraftRGBDHardZClip(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessMinecraftRGBDHardZClip, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3))

    def _observation(self, obs):
        return ProcessMinecraftRGBDHardZClip.process(obs)

    @staticmethod
    def process(frame):
        img = frame.astype(np.float32)
        if frame.size == 240 * 320 * 4:
            img = np.reshape(frame, (240, 320, 4)).astype(np.float32)
            img_color = img[:,:,:3]
            img_depth = img[:,:,3]
        elif frame.size == 240 * 320 * 3:
            assert False, "No depth channel"
        else:
            assert False, "Unknown resolution: {}".format(frame.size)
        img_color_p = np.transpose(img_color, (2, 0, 1))
        img_color_p = img_color_p * (img_depth <= 250).astype(np.float32)
        img_color = np.transpose(img_color_p, (1, 2, 0))
        x_t = cv2.resize(img_color, (84, 84), interpolation=cv2.INTER_AREA)
        assert x_t.shape == (84, 84, 3)
        return x_t.astype(np.uint8)

def wrap_minecraft_84x84_v2(env, frame_skip=1, history_len=1, zclip=False):
  if frame_skip > 1:
    env = SkipFrameEnv(env, skip=frame_skip)
  if zclip:
    env = ProcessMinecraftRGBDHardZClip(env)
  else:
    env = ProcessMinecraftRGBD(env)
  env = Transpose3DEnv(env)
  if history_len > 1:
    env = StackFrameEnv(env, history_len)
  return env
