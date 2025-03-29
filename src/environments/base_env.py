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

from pathlib import Path

from agents import Cop, Thief
from agents.entity import Entity
from maps import Map
from utils import get_thief_category, get_cop_category, get_termination_radius
from utils import ObjectType


class BaseEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
            self,
            map: Map, 
            map_image: Path = None,
            render_mode=None
        ):
        """
        Base class for the environment, based on the Gymnasium environment class (see https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
        Args:
            map: The map object for the environment. Has the agent's initial positions and count.
            render_mode: The mode in which to render the environment (either "human" or "rgb_array")
        """
        super().__init__()

        # Define Space
        self.map = map
        self.map_image = map_image
        self.width, self.height = self.map.window_dimensions
        self.space = pymunk.Space()
        self.map.populate_space(self.space)
        # Define pymunk agents categories. For vision, we want agents to see each other and know their type.
        self.cop_category = get_cop_category()
        self.thief_category = get_thief_category()
        self._termination_radius = get_termination_radius()

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

        # For skrl and MAPPO, we need to define the observation and action spaces
        self.observation_spaces = {
            agent.get_id(): agent.observation_space
            for agent in (self.cops + self.thieves)
        }
        self.action_spaces = {
            agent.get_id(): agent.action_space for agent in (self.cops + self.thieves)
        }
        self.shared_observation_spaces = {}

        max_dim = max(self.map.window_dimensions)
        for team, agents in [("cops", self.cops), ("thieves", self.thieves)]:
            team_space = gym.spaces.Dict(
                {
                    "distance": agents[0].observation_space["distance"],
                    "object_type": agents[0].observation_space["object_type"],
                    "team_positions": gym.spaces.Box(
                        low=0.0, high=max_dim, shape=(len(agents), 2), dtype=np.float16
                    ),
                }
            )
            for agent in agents:
                self.shared_observation_spaces[agent.get_id()] = team_space

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
    def observation_space(self, agent: str):
        return self.agent_name_mapping[agent].observation_space

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

        self.shared_observation_spaces = self.get_shared_observations(observations)

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
        is_terminated = self._termination_criterion()
        # rewards for all agents are placed in the rewards dictionary to be returned
        results = [
            self.agent_name_mapping[agent].step(action[agent], is_terminated)
            for agent in self.agents
        ]
        observations, rewards, terminations, truncations, infos = [
            dict(zip(self.agents, values)) for values in zip(*results)
        ]

        self.shared_observation_spaces = self.get_shared_observations(observations)

        self.state = observations

        self.space.step(1 / 60.0)  # @TODO: Parameterize the time step

        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminations, truncations, infos

    def _get_info(self):
        raise NotImplementedError

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Cops and Robbers")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        draw_options = pymunk.pygame_util.DrawOptions(self.window)
        self.window.fill((255, 255, 255))

        if self.render_mode == "human":

            if self.map_image:
                map_image = pygame.image.load(self.map_image) 
                map_image.set_alpha(int(.75*255))  # Set opacity (0 is fully transparent, 255 is fully opaque)
                map_image = pygame.transform.scale(
                    map_image, 
                    (self.width, self.height)
                )
                self.window.blit(map_image, (0, 0))

            self.space.debug_draw(draw_options)
            
            font = pygame.font.Font(None, 36)
            fps = int(self.clock.get_fps())
            fps_text = font.render(f"FPS: {fps}", True, (0, 0, 0))
            self.window.blit(fps_text, (10, 10))

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

    def _termination_criterion(self):
        """
        Check if the game is over. The game is over if all thieves are caught.
        """
        for thief in self.thieves:
            for cop in self.cops:
                filter = pymunk.ShapeFilter(
                    mask=~(thief.filter_category | cop.filter_category) & 0xFFFFFFFF
                )
                hit = self.space.segment_query_first(
                    thief.body.position, cop.body.position, 1, shape_filter=filter
                )
                if hit is None:
                    if (
                        thief.body.position.get_distance(cop.body.position)
                        < self._termination_radius
                    ):
                        return True

        return False

    def get_shared_observations(self, observations) -> dict:
        """
        Create the actual shared observations based on the observations and agent-specific priority order.
        Args:
            observations (dict): The observations for each agent.
        Returns:
            dict: The shared observations for all agents.
        """
        shared_observations = {}
        cop_positions = np.array(
            [cop.body.position for cop in self.cops], dtype=np.float16
        )
        thief_positions = np.array(
            [thief.body.position for thief in self.thieves], dtype=np.float16
        )

        for team_name, team_agents, team_positions in [
            ("cops", self.cops, cop_positions),
            ("thieves", self.thieves, thief_positions),
        ]:
            if not team_agents:
                continue

            first_agent = team_agents[0]
            obj_masked = (
                np.ones_like(observations[first_agent.get_id()]["object_type"]) * 4
            )
            dist_masked = np.zeros_like(observations[first_agent.get_id()]["distance"])

            for agent in team_agents:
                obs = observations[agent.get_id()]
                obj_types = obs["object_type"]
                distances = obs["distance"]

                for priority_type in agent.observation_priorities:
                    mask = (obj_types == priority_type.value) & (obj_masked == 4)
                    obj_masked[mask] = obj_types[mask]
                    dist_masked[mask] = distances[mask]

            shared_observations[team_name] = {
                "object_type": obj_masked,
                "distance": dist_masked,
                "team_positions": team_positions,
            }

        return shared_observations


def raw_env(map: Map, render_mode="rgb_array") -> AECEnv:
    """
    In support of the PettingZoo API, this function returns a raw environment (see https://pettingzoo.farama.org/api/aec/#about-aec).
    Args:
        map(Map): The map object for the environment
        render_mode(str): The mode in which to render the environment (either "human" or "rgb_array")
    """
    env = BaseEnv(
        map=map,
        render_mode=render_mode,
    )
    env = parallel_to_aec(env)
    return env
