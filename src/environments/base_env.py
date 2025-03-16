from typing import List
import gymnasium as gym
import pymunk
import pygame
import numpy as np

from agents import Cop, Thief
from maps.map import Map


class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cops_count: int, thieves_count: int, map: Map, render_mode=None):
        """
        Base class for the environment, based on the Gymnasium environment class (see https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
        Args:
            cops_count: Number of cops in the environment
            thieves_count: Number of thieves in the environment
            map: The map object for the environment
            render_mode: The mode in which to render the environment (either "human" or "rgb_array")
        """
        super().__init__()
        self.cops: List[Cop] = [Cop() for _ in range(cops_count)]
        self.thieves: List[Thief] = [Thief() for _ in range(thieves_count)]
        self.map = map
        self.width, self.height = self.map.window_dimensions
        self.canvas_width, self.canvas_height = self.map.canvas_dimensions
        self.space = pymunk.Space()
        self.map.populate_space(self.space)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        credit: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#rendering
        """
        self.window = None
        self.clock = None

    def step(self, action):

        if self.render_mode == "human":
            self._render_frame()

        # return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # TODO: Implement the reset method for the agents. Should they be reset to a random position?
        # for cop in self.cops:
        #     cop.reset()
        # for thief in self.thieves:
        #     thief.reset()
        # TODO: Implement the _get_observation and _get_info methods
        # observation = self._get_observation()
        # info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # return observation, info

    def _get_observation(self):
        raise NotImplementedError

    def _get_info(self):
        raise NotImplementedError

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            draw_options = pymunk.pygame_util.DrawOptions(self.window)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        d_canvas = pygame.Surface((self.width, self.height))
        d_canvas.fill((255, 255, 255))
        self.map.render(d_canvas)

        # TODO: Render the agents on the screen

        if self.render_mode == "human":
            self.window.blit(d_canvas, d_canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(60)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(d_canvas)), axes=(1, 0, 2)
            )
