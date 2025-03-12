from typing import List

from agents import Cop, Thief
import gymnasium as gym
import pygame


class BaseEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        cops_count: int,
        thieves_count: int,
    ):
        """
        Base class for the environment
        Args:
            cops_count: Number of cops in the environment
            thieves_count: Number of thieves in the environment
        """
        super().__init__()
        self.cops: List[Cop] = [Cop() for _ in range(cops_count)]
        self.thieves: List[Thief] = [Thief() for _ in range(thieves_count)]

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        for cop in self.cops:
            cop.reset()
        for thief in self.thieves:
            thief.reset()

    def render(self, mode="rgb_array"):
        if mode == "human":
            self._render_human()
        elif mode == "rgb_array":
            return self._render_rgb_array()

    def _render_human(self):
        raise NotImplementedError

    def _render_rgb_array(self):
        raise NotImplementedError
