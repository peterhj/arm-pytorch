import cv2
import gym
import numpy as np

from collections import deque

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class ClippedRewardsWrapper(gym.RewardWrapper):
    def _reward(self, reward):
        """Change all the positive rewards to 1, negative to -1 and keep zero."""
        return np.sign(reward)

class ResetDifficultyWrap(gym.Wrapper):
  def __init__(self, env=None, difficulty=None):
    super(ResetDifficultyWrap, self).__init__(env)
    assert difficulty is not None
    self._difficulty = difficulty

  def _reset(self):
    obs = self.env.reset(difficulty=self._difficulty)
    return obs

  def _step(self, action):
    obs, res, done, info = self.env.step(action)
    return obs, res, done, info

class NumpyObsWrap(gym.Wrapper):
  def __init__(self, env=None):
    super(NumpyObsWrap, self).__init__(env)

  def _reset(self):
    obs = self.env.reset()
    return np.array(obs)

  def _step(self, action):
    obs, res, done, info = self.env.step(action)
    return np.array(obs), res, done, info

class NumpyActionUnwrap(gym.Wrapper):
  def __init__(self, env=None):
    super(NumpyActionUnwrap, self).__init__(env)

  def _reset(self):
    obs = self.env.reset()
    return obs

  def _step(self, action):
    obs, res, done, info = self.env.step(list(action))
    return obs, res, done, info

class ObsZeroMaskWrap(gym.Wrapper):
  def __init__(self, env=None, zero_idxs=[]):
    super(ObsZeroMaskWrap, self).__init__(env)
    self._zero_idxs = zero_idxs.copy()

  def _reset(self):
    obs = self.env.reset()
    obs[self._zero_idxs] = 0.0
    return obs

  def _step(self, action):
    obs, res, done, info = self.env.step(action)
    obs[self._zero_idxs] = 0.0
    return obs, res, done, info

class ObsActionWrap(gym.Wrapper):
  def __init__(self, env=None, default_action=None):
    super(ObsActionWrap, self).__init__(env)
    if default_action is None:
      default_action = np.zeros(env.action_space.shape)
    self._default_action = default_action
    # TODO: observation space.
    act_obs = np.concatenate((np.zeros(env.observation_space.shape), self._default_action), axis=0)
    print("DEBUG:", self.env.observation_space.low[:][0])
    print("DEBUG:", self.env.observation_space.high[:][0])
    self.observation_space = gym.spaces.Box(
        low=env.observation_space.low[:][0],
        high=env.observation_space.high[:][0],
        shape=act_obs.shape)

  def _reset(self):
    obs = self.env.reset()
    act_obs = np.concatenate((obs, self._default_action), axis=0)
    return act_obs

  def _step(self, action):
    obs, res, done, info = self.env.step(action)
    act_obs = np.concatenate((obs, action), axis=0)
    return act_obs, res, done, info

class ObsHistoryConcatWrap(gym.Wrapper):
  def __init__(self, env=None, history_len=1, repeat_reset=False):
    super(ObsHistoryConcatWrap, self).__init__(env)
    assert history_len >= 1
    self._history_len = history_len
    self._repeat_reset = repeat_reset
    self._obs_history = deque([], maxlen=self._history_len)
    # TODO: observation space.
    concat_obs = np.concatenate([np.zeros(self.env.observation_space.shape)] * self._history_len, axis=0)
    print("DEBUG:", self.env.observation_space.low[:][0])
    print("DEBUG:", self.env.observation_space.high[:][0])
    self.observation_space = gym.spaces.Box(
        low=self.env.observation_space.low[:][0],
        high=self.env.observation_space.high[:][0],
        shape=concat_obs.shape)

  def _reset(self):
    self._obs_history.clear()
    obs = self.env.reset()
    if self._repeat_reset:
      while len(self._obs_history) < self._history_len:
        self._obs_history.append(obs)
    else:
      while len(self._obs_history) < self._history_len - 1:
        self._obs_history.append(np.zeros(obs.shape))
      self._obs_history.append(obs)
    assert len(self._obs_history) == self._history_len
    obs_hist = np.concatenate(list(self._obs_history), axis=0)
    return obs_hist

  def _step(self, action):
    obs, res, done, info = self.env.step(action)
    self._obs_history.append(obs)
    assert len(self._obs_history) == self._history_len
    obs_hist = np.concatenate(list(self._obs_history), axis=0)
    return obs_hist, res, done, info

class FlatObsWrap(gym.Wrapper):
  def __init__(self, env=None):
    super(FlatObsWrap, self).__init__(env)
    # TODO: observation space.
    flat_obs = np.ravel(np.zeros(self.env.observation_space.shape))
    print("DEBUG:", self.env.observation_space.low[:][0])
    print("DEBUG:", self.env.observation_space.high[:][0])
    self.observation_space = gym.spaces.Box(
        low=self.env.observation_space.low[:][0],
        high=self.env.observation_space.high[:][0],
        shape=flat_obs.shape)

  def _reset(self):
    obs = self.env.reset()
    return np.ravel(obs)

  def _step(self, action):
    obs, res, done, info = self.env.step(action)
    return np.ravel(obs), res, done, info

class SkipFrameEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(SkipFrameEnv, self).__init__(env)
        self._skip = skip

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        obs = self.env.reset()
        return obs

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not belive how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

class StackFrameEnv(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        super(StackFrameEnv, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0] * k, shp[1], shp[2]))

    def _reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.array(LazyFrames(list(self.frames)))

class DelayedActionEnv(gym.Wrapper):
    def __init__(self, env=None, act_delay=None):
        super(DelayedActionEnv, self).__init__(env)
        assert act_delay is not None and act_delay >= 1
        self._act_delay = act_delay
        self._act_queue = deque([], maxlen=act_delay)
        self._reset_obs = None

    def _reset(self):
        obs = self.env.reset()
        self._act_queue.clear()
        self._reset_obs = obs
        return obs

    def _step(self, action):
        if len(self._act_queue) == self._act_delay:
            delayed_action = self._act_queue.popleft()
            obs, reward, done, info = self.env.step(delayed_action)
        else:
            obs = self._reset_obs
            reward = 0.0
            done = False
            info = None
        self._act_queue.append(action)
        return obs, reward, done, info

class Transpose3DEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(Transpose3DEnv, self).__init__(env)
        prev_space = env.observation_space
        new_shape = (prev_space.shape[2], prev_space.shape[0], prev_space.shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape)

    def _observation(self, obs):
        return Transpose3DEnv.process(obs)

    @staticmethod
    def process(frame):
        frame = np.transpose(frame, axes=[2, 0, 1])
        return frame

class FlatActionObsEnv(gym.Wrapper):
    def __init__(self, env):
        super(FlatActionObsEnv, self).__init__(env)
        obs_size = np.squeeze(env.observation_space.shape)
        act_size = np.squeeze(env.action_space.shape)
        assert obs_size.shape == ()
        assert act_size.shape == ()
        new_obs_size = np.array(act_size[0] + obs_size[0])
        # TODO
        #self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k))
        self._observation_space = env.observation_space
        self._observation_space.shape = new_obs_size
        self._obs_size = obs_size
        self._act_size = act_size

    def _get_nil_act(self):
        # TODO
        #raise NotImplementedError
        return np.zeros(self._act_size)

    def _combine_act_obs(self, action, next_obs):
        # TODO
        #raise NotImplementedError
        return np.concatenate((action, next_obs))

    def _reset(self):
        obs = self.env.reset()
        return self._combine_act_obs(self._get_nil_act(), next_obs)

    def _step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return self._combine_act_obs(action, next_obs), reward, done, info

class ConcatFlatObsHistoryEnv(gym.Wrapper):
    def __init__(self, env, history_len):
        super(ConcatFlatObsHistoryEnv, self).__init__(env)
        # TODO
        self._history_len = history_len
        self._history = deque([], maxlen=history_len)

    def _get_obs_history(self):
        assert len(self._history) > 0
        if len(self._history) < self._history_len:
            last_obs = self._history.popleft()
            while len(self._history) < self._history_len:
                self._history.appendleft(last_obs)
        # TODO
        #raise NotImplementedError
        return np.concatenate(list(self._history))

    def _reset(self):
        obs = self.env.reset()
        self._history.append(obs)
        return self._get_obs_history()

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._history.append(obs)
        return self._get_obs_history(), reward, done, info

class PreprocAtariPongMask(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreprocAtariPongMask, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(210, 160, 3))

    def _observation(self, obs):
        if obs.size == 210 * 160 * 3:
            img = np.reshape(obs, (210, 160, 3))
        else:
            assert False, "Unknown resolution: {}".format(obs.size)
        img[34:194,55:105,0] = 144
        img[34:194,55:105,1] = 72
        img[34:194,55:105,2] = 17
        return img

class PreprocAtariBreakoutMask(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreprocAtariBreakoutMask, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(210, 160, 3))

    def _observation(self, obs):
        if obs.size == 210 * 160 * 3:
            img = np.reshape(obs, (210, 160, 3))
        else:
            assert False, "Unknown resolution: {}".format(obs.size)
        img[105:175,:,:] = 0
        return img

class ProcessAtariGray84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessAtariGray84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def _observation(self, obs):
        if obs.size == 210 * 160 * 3:
            img = np.reshape(obs, (210, 160, 3)).astype(np.float32)
        else:
            assert False, "Unknown resolution: {}".format(obs.size)
        x_t = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        x_t = cv2.resize(x_t, (84, 84), interpolation=cv2.INTER_AREA)
        x_t = np.reshape(x_t, (84, 84, 1))
        return x_t.astype(np.uint8)

class ProcessAtariGray84Flicker(gym.ObservationWrapper):
    def __init__(self, env=None, flicker_prob=None):
        super(ProcessAtariGray84Flicker, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1))
        self._flicker_prob = flicker_prob

    def _observation(self, obs):
        r = np.random.choice(2)
        if r == 0:
            if obs.size == 210 * 160 * 3:
                img = np.reshape(obs, (210, 160, 3)).astype(np.float32)
            else:
                assert False, "Unknown resolution: {}".format(obs.size)
            x_t = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            x_t = cv2.resize(x_t, (84, 84), interpolation=cv2.INTER_AREA)
            x_t = np.reshape(x_t, (84, 84, 1))
            return x_t.astype(np.uint8)
        elif r == 1:
            x_t = np.zeros((84, 84, 1), dtype=np.uint8)
            return x_t
        else:
            assert False

class ProcessAtariGray84Repeat(gym.Wrapper):
    def __init__(self, env=None, repeat_prob=None):
        super(ProcessAtariGray84Repeat, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1))
        self._repeat_prob = repeat_prob
        self._prev_obs = None
        self._curr_obs = None

    def _reset(self):
        self._prev_obs = np.zeros((84, 84, 1), dtype=np.uint8)
        obs = self.env.reset()
        r = np.random.choice(2)
        if r == 0:
            self._prev_obs = obs
            self._curr_obs = obs
        elif r == 1:
            self._curr_obs = self._prev_obs
        else:
            assert False
        return self._process_obs(self._curr_obs)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        r = np.random.choice(2)
        if r == 0:
            self._prev_obs = obs
            self._curr_obs = obs
        elif r == 1:
            self._curr_obs = self._prev_obs
        else:
            assert False
        return self._process_obs(self._curr_obs), reward, done, info

    def _process_obs(self, obs):
        if obs.size == 210 * 160 * 3:
            img = np.reshape(obs, (210, 160, 3)).astype(np.float32)
        else:
            assert False, "Unknown resolution: {}".format(obs.size)
        x_t = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        x_t = cv2.resize(x_t, (84, 84), interpolation=cv2.INTER_AREA)
        x_t = np.reshape(x_t, (84, 84, 1))
        return x_t.astype(np.uint8)

def wrap_atari_84x84_v2(env, frame_skip=4, preproc_pong_mask=False, preproc_breakout_mask=False, frame_flicker=False, frame_repeat=False, history_len=4):
  assert 'NoFrameskip' in env.spec.id
  env = NoopResetEnv(env, noop_max=30)
  env = EpisodicLifeEnv(env)
  if frame_skip > 1:
    env = SkipFrameEnv(env, skip=frame_skip)
  if 'FIRE' in env.unwrapped.get_action_meanings():
    env = FireResetEnv(env)
  if preproc_pong_mask:
    env = PreprocAtariPongMask(env)
  elif preproc_breakout_mask:
    env = PreprocAtariBreakoutMask(env)
  if frame_flicker:
    env = ProcessAtariGray84Flicker(env, 0.5)
  elif frame_repeat:
    env = ProcessAtariGray84Repeat(env, 0.5)
  else:
    env = ProcessAtariGray84(env)
  env = Transpose3DEnv(env)
  if history_len > 1:
    env = StackFrameEnv(env, history_len)
  env = ClippedRewardsWrapper(env)
  return env

def wrap_flat(env, combined_action_obs=False, history_len=1):
  if combined_action_obs:
    env = FlatActionObsEnv(env)
  if history_len > 1:
    env = ConcatFlatObsHistoryEnv(env, history_len)
  return env
