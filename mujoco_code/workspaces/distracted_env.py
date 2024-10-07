from abc import ABC, abstractmethod

import gym
import numpy as np


class AbstractDistractedWrapper(gym.Wrapper, ABC):
    def __init__(
        self, env, distraction_dims: int, distraction_scale: float = 1.0
    ) -> None:
        super().__init__(env)
        assert distraction_dims > 0
        self.d = distraction_dims
        self.scale = distraction_scale
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.reset().shape, dtype=np.float32
        )

    @abstractmethod
    def _get_distract_obs(self):
        raise NotImplementedError

    def reset(self):
        obs = self.env.reset()
        return np.concatenate([obs, self._get_distract_obs()])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return np.concatenate([obs, self._get_distract_obs()]), rew, done, info


class GaussianDistractedWrapper(AbstractDistractedWrapper):
    def _get_distract_obs(self):
        return self.scale * np.random.normal(size=(self.d,))


class GaussianMixtureDistractedWrapper(AbstractDistractedWrapper):
    def _get_distract_obs(self):
        o1 = self.scale * np.random.normal(loc=0.0, size=(self.d,))
        o2 = self.scale * np.random.normal(loc=1.0, size=(self.d,))
        return o1 + o2


class InterleavedGaussianMixtureDistractedWrapper(AbstractDistractedWrapper):
    def __init__(
        self, env, distraction_dims: int, distraction_scale: float = 1.0
    ) -> None:
        assert distraction_dims % 2 == 0
        super().__init__(env, distraction_dims, distraction_scale)

    def _get_distract_obs(self):
        o1 = self.scale * np.random.normal(loc=0.0, size=(self.d // 2,))
        o2 = self.scale * np.random.normal(loc=1.0, size=(self.d // 2,))

        interleaved = np.empty((self.d,))
        interleaved[0::2] = o1
        interleaved[1::2] = o2

        return interleaved
