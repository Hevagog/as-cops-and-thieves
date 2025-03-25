from typing import List
import functools
import pymunk
import pygame
import gymnasium as gym
import numpy as np
import itertools
from pettingzoo import ParallelEnv, AECEnv
from pettingzoo.utils import parallel_to_aec
from gymnasium.utils import seeding


from agents import Cop, Thief
from agents.entity import Entity
from maps import Map
from utils import get_thief_category, get_cop_category


class BaseEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, map: Map, render_mode=None):
        """
        Base class for the environment, based on the Gymnasium environment class (see https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
        Args:
            map: The map object for the environment. Has the agent's initial positions and count.
            render_mode: The mode in which to render the environment (either "human" or "rgb_array")
        """
        super().__init__()

        # Define Space
        self.map = map
        self.width, self.height = self.map.window_dimensions
        self.space = pymunk.Space()
        self.map.populate_space(self.space)
        # Define pymunk agents categories. For vision, we want agents to see each other and know their type.
        self.cop_category = get_cop_category()
        self.thief_category = get_thief_category()

        group_counter = itertools.count(1)
        # Create the agents
        self.cops: List[Cop] = [
            Cop(
                start_position=pymunk.Vec2d(*map.cops_positions[id]),
                space=self.space,
                group=next(group_counter),
                id=f"cop_{id}",
                filter_category=self.cop_category,
            )
            for id in range(map.cops_count)
        ]
        self.thieves: List[Thief] = [
            Thief(
                start_position=pymunk.Vec2d(*map.thieves_positions[id]),
                space=self.space,
                group=next(group_counter),
                id=f"thief_{id}",
                filter_category=self.thief_category,
            )
            for id in range(map.thieves_count)
        ]

        # possible_agents is a static list of all possible agents in the environment that could ever be present in the environment.
        # This generalization is needed for the PettingZoo API.
        self.possible_agents = [agent.get_id() for agent in (self.cops + self.thieves)]
        self.agent_name_mapping = {
            agent.get_id(): agent for agent in (self.cops + self.thieves)
        }

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

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: Entity):
        return agent.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.spaces.Discrete:
        return self.agent_name_mapping[agent].action_space

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ):
        # Copied from gym.Env.reset method
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)

        self.agents = self.possible_agents[:]
        for agent in self.agents:
            self.agent_name_mapping[agent].reset()

        observations = {
            agent: self.agent_name_mapping[agent].get_observation()
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}

        self.state = observations

        if self.render_mode == "human":
            self._render_frame()

        return observations, infos

    def step(self, action):
        """
        Method to update the environment's state.
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not action:
            self.agents = []
            return {}, {}, {}, {}, {}

        # rewards for all agents are placed in the rewards dictionary to be returned
        results = [
            self.agent_name_mapping[agent].step(action[agent]) for agent in self.agents
        ]
        observations, rewards, terminations, truncations, infos = [
            dict(zip(self.agents, values)) for values in zip(*results)
        ]

        self.state = observations

        self.space.step(1 / 60.0)  # @TODO: Parameterize the time step

        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, truncations, False, infos

    def _get_info(self):
        raise NotImplementedError

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        draw_options = pymunk.pygame_util.DrawOptions(self.window)
        self.window.fill((255, 255, 255))
        agents_positions = [
            self.agent_name_mapping[agent].body.position for agent in self.agents
        ]

        if self.render_mode == "human":
            self.space.debug_draw(draw_options)
            pygame.display.flip()
            self.clock.tick(60)
        else:  # rgb_array
            self.space.debug_draw(draw_options)
            return pygame.surfarray.array3d(self.window)

        self.clock.tick(60)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

    def observe(self, agent):
        return np.array(agent.get_observation(), dtype=np.float32)


def raw_env(map: Map, cops_count=1, thieves_count=1, render_mode="rgb_array") -> AECEnv:
    """
    In support of the PettingZoo API, this function returns a raw environment (see https://pettingzoo.farama.org/api/aec/#about-aec).
    Args:
        map(Map): The map object for the environment
        cops_count(int): Number of cops in the environment
        thieves_count(int): Number of thieves in the environment
        render_mode(str): The mode in which to render the environment (either "human" or "rgb_array")
    """
    env = BaseEnv(
        cops_count=cops_count,
        thieves_count=thieves_count,
        map=map,
        render_mode=render_mode,
    )
    env = parallel_to_aec(env)
    return env
