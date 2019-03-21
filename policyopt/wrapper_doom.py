#!/usr/bin/env python3.5

from policyopt.wrapper import *

import cv2
import gym
import numpy as np
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

from collections import deque

class ProcessFrame84x84Doom(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84x84Doom, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3))

    def _observation(self, obs):
        return ProcessFrame84x84Doom.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 120 * 160 * 3:
            img = np.reshape(frame, [120, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        #img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        #resized_screen = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        x_t = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR)
        #x_t = resized_screen[18:102, :]
        #x_t = np.reshape(x_t, [84, 84, 1])
        assert x_t.shape == (84, 84, 3)
        #x_t = np.transpose(x_t, axes=[2, 0, 1])
        #assert x_t.shape == (3, 84, 84)
        return x_t.astype(np.uint8)

def wrap_doom_84x84(env, frame_skip=4, history_len=4):
  env = ToDiscrete("minimal")(env)
  env = SkipFrameEnv(env, skip=frame_skip)
  env = ProcessFrame84x84Doom(env)
  env = Transpose3DEnv(env)
  if history_len > 1:
    env = StackFrameEnv(env, history_len)
  return env

class PreprocDoomRGBDMask(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreprocDoomRGBDMask, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 120, 160))

    def _observation(self, obs):
        if obs.size == 120 * 160 * 4:
            img = np.reshape(obs, (4, 120, 160))
        else:
            assert False, "Unknown resolution: {}".format(obs.size)
        img[:,30:90,50:110] = 0
        return img

class PreprocDoomRGBDMaskBig(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreprocDoomRGBDMaskBig, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 120, 160))

    def _observation(self, obs):
        if obs.size == 120 * 160 * 4:
            img = np.reshape(obs, (4, 120, 160))
        else:
            assert False, "Unknown resolution: {}".format(obs.size)
        img[:,10:110,30:130] = 0
        return img

class PreprocDoomRGBDMaskOut(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreprocDoomRGBDMaskOut, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 120, 160))

    def _observation(self, obs):
        if obs.size == 120 * 160 * 4:
            img = np.reshape(obs, (4, 120, 160))
        else:
            assert False, "Unknown resolution: {}".format(obs.size)
        img[:,:40,:] = 0
        img[:,80:,:] = 0
        img[:,:,:60] = 0
        img[:,:,100:] = 0
        return img

class ProcessDoomRGBD1x1(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessDoomRGBD1x1, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 1, 1))

    def _observation(self, obs):
        return ProcessDoomRGBD1x1.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 120 * 160 * 4:
            img = np.reshape(frame, (4, 120, 160)).astype(np.float32)
            img_color = img[:3,:,:]
        else:
            assert False, "Unknown resolution: {}".format(frame.size)
        img_color = np.transpose(img_color, (1, 2, 0))
        x_t = cv2.resize(img_color, (1, 1), interpolation=cv2.INTER_AREA)
        assert x_t.shape == (1, 1, 3)
        x_t = np.reshape(x_t, (3, 1, 1))
        return x_t.astype(np.uint8)

def wrap_doom_1x1_v2(env, frame_skip=4, history_len=4, zclip=False):
  env = ToDiscrete("minimal")(env)
  if frame_skip > 1:
    env = SkipFrameEnv(env, skip=frame_skip)
  if zclip:
    raise NotImplementedError
  else:
    env = ProcessDoomRGBD1x1(env)
  if history_len > 1:
    env = StackFrameEnv(env, history_len)
  return env

class ProcessDoomRGBD20(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessDoomRGBD20, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(20, 20, 3))

    def _observation(self, obs):
        return ProcessDoomRGBD20.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 120 * 160 * 4:
            img = np.reshape(frame, (4, 120, 160)).astype(np.float32)
            img_color = img[:3,:,:]
        else:
            assert False, "Unknown resolution: {}".format(frame.size)
        img_color = np.transpose(img_color, (1, 2, 0))
        x_t = cv2.resize(img_color, (20, 20), interpolation=cv2.INTER_AREA)
        assert x_t.shape == (20, 20, 3)
        return x_t.astype(np.uint8)

def wrap_doom_20x20_v2(env, frame_skip=4, action_delay=None, zclip=False, depth=False, frame_flicker=False, history_len=4):
  env = ToDiscrete("minimal")(env)
  if frame_skip > 1:
    env = SkipFrameEnv(env, skip=frame_skip)
  if action_delay is not None and action_delay >= 1:
    raise NotImplementedError
  if zclip:
    raise NotImplementedError
  elif depth:
    raise NotImplementedError
  elif frame_flicker:
    raise NotImplementedError
  else:
    env = ProcessDoomRGBD20(env)
  env = Transpose3DEnv(env)
  if history_len > 1:
    env = StackFrameEnv(env, history_len)
  return env

class ProcessDoomRGBD44(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessDoomRGBD44, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(44, 44, 3))

    def _observation(self, obs):
        return ProcessDoomRGBD44.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 120 * 160 * 4:
            img = np.reshape(frame, (4, 120, 160)).astype(np.float32)
            img_color = img[:3,:,:]
        else:
            assert False, "Unknown resolution: {}".format(frame.size)
        img_color = np.transpose(img_color, (1, 2, 0))
        x_t = cv2.resize(img_color, (44, 44), interpolation=cv2.INTER_AREA)
        assert x_t.shape == (44, 44, 3)
        return x_t.astype(np.uint8)

def wrap_doom_44x44_v2(env, frame_skip=4, history_len=4, zclip=False):
  env = ToDiscrete("minimal")(env)
  if frame_skip > 1:
    env = SkipFrameEnv(env, skip=frame_skip)
  if zclip:
    raise NotImplementedError
  else:
    env = ProcessDoomRGBD44(env)
  env = Transpose3DEnv(env)
  if history_len > 1:
    env = StackFrameEnv(env, history_len)
  return env

class ProcessDoomRGBD(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessDoomRGBD, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3))

    def _observation(self, obs):
        return ProcessDoomRGBD.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 120 * 160 * 4:
            img = np.reshape(frame, (4, 120, 160)).astype(np.float32)
            img_color = img[:3,:,:]
        else:
            assert False, "Unknown resolution: {}".format(frame.size)
        img_color = np.transpose(img_color, (1, 2, 0))
        x_t = cv2.resize(img_color, (84, 84), interpolation=cv2.INTER_LINEAR)
        assert x_t.shape == (84, 84, 3)
        return x_t.astype(np.uint8)

class ProcessDoomRGBDDepth(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessDoomRGBDDepth, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def _observation(self, obs):
        return ProcessDoomRGBDDepth.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 120 * 160 * 4:
            img = np.reshape(frame, (4, 120, 160)).astype(np.float32)
            img_depth = img[3,:,:]
        else:
            assert False, "Unknown resolution: {}".format(frame.size)
        #img_depth = np.reshape(img_depth, (120, 160, 1))
        x_t = cv2.resize(img_depth, (84, 84), interpolation=cv2.INTER_LINEAR)
        assert x_t.shape == (84, 84)
        x_t = np.reshape(x_t, (84, 84, 1))
        return x_t.astype(np.uint8)

class ProcessDoomRGBDHardZClip(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessDoomRGBDHardZClip, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3))

    def _observation(self, obs):
        return ProcessDoomRGBDHardZClip.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 120 * 160 * 4:
            img = np.reshape(frame, (4, 120, 160)).astype(np.float32)
            img_color = img[:3,:,:]
            img_depth = img[3,:,:]
        else:
            assert False, "Unknown resolution: {}".format(frame.size)
        img_color = img_color * (img_depth <= 14).astype(np.float32)
        img_color = np.transpose(img_color, (1, 2, 0))
        x_t = cv2.resize(img_color, (84, 84), interpolation=cv2.INTER_LINEAR)
        assert x_t.shape == (84, 84, 3)
        return x_t.astype(np.uint8)

class ProcessDoomRGBDFlicker(gym.ObservationWrapper):
    def __init__(self, env=None, flicker_prob=None):
        super(ProcessDoomRGBDFlicker, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3))

    def _observation(self, obs):
        r = np.random.choice(2)
        if r == 0:
            if obs.size == 120 * 160 * 4:
                img = np.reshape(obs, (4, 120, 160)).astype(np.float32)
                img_color = img[:3,:,:]
            else:
                assert False, "Unknown resolution: {}".format(obs.size)
            img_color = np.transpose(img_color, (1, 2, 0))
            x_t = cv2.resize(img_color, (84, 84), interpolation=cv2.INTER_LINEAR)
            assert x_t.shape == (84, 84, 3)
            return x_t.astype(np.uint8)
        elif r == 1:
            return np.zeros((84, 84, 3), dtype=np.uint8)
        else:
            assert False

class ProcessDoomRGBDRepeat(gym.Wrapper):
    def __init__(self, env=None, frame_repeat=None):
        super(ProcessDoomRGBDRepeat, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3))
        self._frame_repeat = frame_repeat
        #self._obs_queue = deque([], maxlen=frame_repeat)
        self._frame_ctr = 0
        self._curr_obs = None

    def _reset(self):
        obs = self.reset()
        self._frame_ctr = 0
        self._curr_obs = obs
        return self._process_obs(self._curr_obs)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frame_ctr += 1
        if self._frame_ctr == self._frame_repeat:
            self._frame_ctr = 0
            self._curr_obs = obs
        return self._process_obs(self._curr_obs), reward, done, info

    def _process_obs(self, obs):
        if obs.size == 120 * 160 * 4:
            img = np.reshape(obs, (4, 120, 160)).astype(np.float32)
            img_color = img[:3,:,:]
        else:
            assert False, "Unknown resolution: {}".format(obs.size)
        img_color = np.transpose(img_color, (1, 2, 0))
        x_t = cv2.resize(img_color, (84, 84), interpolation=cv2.INTER_LINEAR)
        assert x_t.shape == (84, 84, 3)
        return x_t.astype(np.uint8)

def wrap_doom_84x84_v2(env, frame_skip=4, action_delay=None, preproc_mask=False, preproc_mask_big=False, preproc_mask_out=False, zclip=False, depth=False, frame_flicker=False, frame_repeat=None, history_len=4):
  env = ToDiscrete("minimal")(env)
  if frame_skip > 1:
    env = SkipFrameEnv(env, skip=frame_skip)
  if action_delay is not None and action_delay > 0:
    env = DelayedActionEnv(env, action_delay)
  if preproc_mask:
    env = PreprocDoomRGBDMask(env)
  elif preproc_mask_big:
    env = PreprocDoomRGBDMaskBig(env)
  elif preproc_mask_out:
    env = PreprocDoomRGBDMaskOut(env)
  if zclip:
    env = ProcessDoomRGBDHardZClip(env)
  elif depth:
    env = ProcessDoomRGBDDepth(env)
  elif frame_flicker:
    env = ProcessDoomRGBDFlicker(env, 0.5)
  elif frame_repeat is not None and frame_repeat > 1:
    env = ProcessDoomRGBDRepeat(env, frame_repeat)
  else:
    env = ProcessDoomRGBD(env)
  env = Transpose3DEnv(env)
  if history_len > 1:
    env = StackFrameEnv(env, history_len)
  return env
