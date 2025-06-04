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
import random  # Add random for region selection if np_random is not yet available

from pathlib import Path

from agents import Cop, Thief
from agents.entity import Entity
from maps import Map
from utils import (
    get_thief_category,
    get_cop_category,
    get_termination_radius,
    sample_spawn_position,
)
from environments.observation_spaces import (
    _init_shared_observation_space,
    get_shared_observations,
)


class BaseEnv(ParallelEnv):
    """
    Base environment for a multi-agent cops and thieves pursuit scenario.

    This environment implements the PettingZoo ParallelEnv interface for multi-agent reinforcement
    learning. It simulates a pursuit-evasion game where cops attempt to catch thieves in a 2D space
    with physical constraints managed through pymunk.

    The environment handles:
    - Agent observation and action spaces
    - Physics-based movement and collisions
    - Agent-specific partial observations via raycasting
    - Team-based information sharing
    - Termination conditions and rewards

    Attributes:
        metadata (dict): Supported render modes for visualization
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        map: Map,
        map_image: Path = None,
        render_mode=None,
        max_step_count: int = 400,
        time_step: float = 1 / 15.0,
    ):
        """
        Initialize the cops and thieves environment with map configuration and rendering options.

        Args:
            map (Map): The map object defining the environment layout, obstacles, and agent positions.
            map_image (Path, optional): Path to an image file for background rendering.
            render_mode (str, optional): Visualization mode, either "human" for interactive display
                                        or "rgb_array" for programmatic access.
            max_step_count (int, optional): Maximum number of steps before environment truncation.
                                          Defaults to 400.
            time_step (float, optional): Time step for physics simulation. Defaults to 1/60.0.
        """
        super().__init__()

        # Define Space
        self.map = map
        self.map_image = map_image
        self.width, self.height = self.map.window_dimensions
        self.space = pymunk.Space()
        self.map.populate_space(self.space)

        self.step_count = 0
        self.max_step_count = max_step_count
        self.time_step = time_step

        # Define pymunk agents categories. For vision, we want agents to see each other and know their type.
        self.cop_category = get_cop_category()
        self.thief_category = get_thief_category()
        self._termination_radius = get_termination_radius()

        # Create the agents
        group_counter = itertools.count(1)
        self.cops: List[Cop] = self._init_cops(group_counter)
        self.thieves: List[Thief] = self._init_thieves(group_counter)

        # possible_agents is a static list of all possible agents in the environment that could ever be present in the environment.
        # This generalization is needed for the PettingZoo API.
        self.possible_agents = [agent.get_id() for agent in (self.cops + self.thieves)]
        self.agent_name_mapping = {
            agent.get_id(): agent for agent in (self.cops + self.thieves)
        }

        # For skrl and MAPPO, we need to define the observation and action spaces
        self.observation_spaces = {
            agent.get_id(): agent.observation_space
            for agent in (self.cops + self.thieves)
        }
        self.action_spaces = {
            agent.get_id(): agent.action_space for agent in (self.cops + self.thieves)
        }
        # We have to define state spaces and shared observation spaces for all agents.
        shared_obs_space = _init_shared_observation_space(
            map=self.map, cops=self.cops, thieves=self.thieves
        )
        self.shared_observation_spaces = shared_obs_space
        self._shared_observation_spaces = shared_obs_space
        self.state_space = shared_obs_space

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._init_rendering()

    def _get_non_colliding_position(
        self, regions: List[dict], entity: Entity, max_attempts=20
    ):
        if not regions:
            print(
                f"Error: No spawn regions provided for agent {entity.get_id()}. Cannot find position."
            )
            # Fallback: return a default position or raise an error.
            # For now, let's attempt to use the entity's current position or a map center.
            # This case should ideally be prevented by ensuring agents always have valid spawn regions.
            return (
                entity.body.position
                if entity.body
                else (self.width / 2, self.height / 2)
            )

        # Randomly select one region from the list
        if hasattr(self, "_np_random") and self._np_random:
            # Create a new list of regions for choice because NumPy's choice
            # might have issues with lists of dictionaries directly depending on version/usage.
            # Or, more simply, choose an index.
            selected_region_index = self._np_random.integers(len(regions))
            selected_region = regions[selected_region_index]
        else:
            selected_region = random.choice(
                regions
            )  # Fallback if np_random not initialized

        for _ in range(max_attempts):
            pos = sample_spawn_position(selected_region)
            # Check if the entity's bounding box would collide at the sampled position
            query_info = self.space.point_query_nearest(
                pos, entity.get_radius(), entity.ray_filter
            )
            if not query_info:  # No collision detected
                return pos

        print(
            f"Warning: Could not find non-colliding position for {entity.get_id()} in selected region {selected_region} after {max_attempts} attempts. Defaulting to region center."
        )
        return (
            selected_region["x"] + selected_region["w"] / 2,
            selected_region["y"] + selected_region["h"] / 2,
        )

    def _init_cops(self, group_counter) -> List[Cop]:
        """
        Initialize the cops in the environment.

        Places cops at their designated starting positions from the map configuration.
        Each cop is assigned a unique group identifier for collision filtering.

        Args:
            group_counter (iterator): Counter for assigning unique group IDs to agents

        Returns:
            List[Cop]: List of initialized cop agents
        """
        return [
            Cop(
                start_position=pymunk.Vec2d(*self.map.cops_positions[id]),
                space=self.space,
                group=next(group_counter),
                id=f"cop_{id}",
                filter_category=self.cop_category,
            )
            for id in range(self.map.cops_count)
        ]

    def _init_thieves(self, group_counter) -> List[Thief]:
        """
        Initialize the thieves in the environment.

        Places thieves at their designated starting positions from the map configuration.
        Each thief is assigned a unique group identifier for collision filtering.

        Args:
            group_counter (iterator): Counter for assigning unique group IDs to agents

        Returns:
            List[Thief]: List of initialized thief agents
        """
        return [
            Thief(
                start_position=pymunk.Vec2d(*self.map.thieves_positions[id]),
                space=self.space,
                group=next(group_counter),
                id=f"thief_{id}",
                filter_category=self.thief_category,
            )
            for id in range(self.map.thieves_count)
        ]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gym.Space:
        """
        Get the observation space for a specific agent.

        Args:
            agent (str): The agent's unique identifier

        Returns:
            gym.Space: The agent's observation space
        """
        return self.agent_name_mapping[agent].observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.spaces:
        """
        Get the action space for a specific agent.

        Args:
            agent (str): The agent's unique identifier

        Returns:
            gym.spaces: The agent's action space
        """
        return self.agent_name_mapping[agent].action_space

    @functools.lru_cache(maxsize=None)
    def get_base_observation_space_structure(self) -> gym.spaces:
        """
        Get the static base structure of observation spaces for all agents.

        This returns the initial definition of observation spaces without nesting
        or modifications that happen during runtime.

        Returns:
            gym.spaces.Dict: Dictionary of observation spaces by agent ID
        """
        return self._shared_observation_spaces

    @functools.lru_cache(maxsize=None)
    def get_nested_agent_observation_spaces(self) -> gym.spaces:
        """
        Get a flattened version of agent observation spaces suitable for MAPPO.

        Instead of deep nesting, this creates a flat structure where each agent's
        space includes their own observations plus flattened versions of others'.

        Returns:
            gym.spaces.Dict: Dictionary of flattened observation spaces
        """
        flat_spaces = {}

        for agent_id in self._shared_observation_spaces:
            # Start with agent's own space
            agent_space = {}
            for key, space in self._shared_observation_spaces[agent_id].spaces.items():
                agent_space[key] = space

            # Add flattened versions of other agents' spaces
            for other_id in self._shared_observation_spaces:
                if other_id != agent_id:
                    for key, space in self._shared_observation_spaces[
                        other_id
                    ].spaces.items():
                        agent_space[f"{other_id}_{key}"] = space

            flat_spaces[agent_id] = gym.spaces.Dict(agent_space)

        return gym.spaces.Dict(flat_spaces)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """
        Reset the environment to an initial state.

        Resets all agents to their starting positions and regenerates observations.

        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional configuration options

        Returns:
            tuple: (observations, infos)
                - observations: Dictionary of initial observations for each agent
                - infos: Dictionary of additional information for each agent
        """

        # Copied from gym.Env.reset method
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)
        else:  # Ensure _np_random is initialized even if seed is None
            if not hasattr(self, "_np_random") or self._np_random is None:
                self._np_random, self._np_random_seed = seeding.np_random(None)

        self.agents = self.possible_agents[:]
        for agent_id in self.agents:
            agent_entity = self.agent_name_mapping[agent_id]
            if agent_id in self.map.agent_spawn_regions:
                spawn_regions_for_agent = self.map.agent_spawn_regions[agent_id]
                if spawn_regions_for_agent:  # Ensure the list is not empty
                    pos = self._get_non_colliding_position(
                        spawn_regions_for_agent, agent_entity
                    )
                    agent_entity.reset(pos)
                else:
                    print(
                        f"Warning: Agent {agent_id} has an empty list of spawn regions. Using default reset (current position or initial)."
                    )
                    agent_entity.reset()  # Agent resets to its own default
            else:
                print(
                    f"Warning: No spawn regions defined in map for agent {agent_id}. Using default reset (current position or initial)."
                )
                agent_entity.reset()  # Agent resets to its own default

        observations = {
            agent_id: self.agent_name_mapping[
                agent_id
            ].get_observation()  # Use agent_id
            for agent_id in self.agents
        }
        infos = {agent_id: {} for agent_id in self.agents}  # Use agent_id

        self.shared_observation_spaces = get_shared_observations(
            observations, self.cops, self.thieves
        )

        if self.render_mode == "human":
            pygame.event.clear()
            self._render_frame()

        self.step_count = 0

        return observations, infos

    def step(self, action) -> tuple[dict, dict, dict, dict, dict]:
        """
        Update the environment state based on agent actions.

        Processes all agents' actions simultaneously, updates the physics simulation,
        calculates rewards, and determines if the episode should terminate.

        Args:
            action (dict): Dictionary mapping agent IDs to their selected actions

        Returns:
            tuple: (observations, rewards, terminations, truncations, infos)
                - observations: New observations after taking actions
                - rewards: Rewards received by each agent
                - terminations: Whether each agent has reached a terminal state
                - truncations: Whether each agent has reached a truncation state
                - infos: Additional information for each agent
        """
        self.step_count += 1
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
        observations, rewards, terminations, _, infos = [
            dict(zip(self.agents, values)) for values in zip(*results)
        ]

        self.shared_observation_spaces = get_shared_observations(
            observations, self.cops, self.thieves
        )

        self.space.step(self.time_step)

        if self.render_mode == "human":
            self._render_frame()

        truncations = {agent: is_terminated[1] for agent in self.agents}

        winner = None

        if any(terminations.values()):
            self.agents = []
            if is_terminated[0]:
                winner = "cop"
            else:
                winner = "thief"

        # print(rewards)

        for agent in infos:
            infos[agent]["winner"] = winner

        return observations, rewards, terminations, truncations, infos

    def state(self) -> dict:
        """
        Get the complete environment state.

        Returns the shared observation spaces which contain all relevant
        state information including agent positions and raycast data.

        Returns:
            dict: The full environment state
        """
        return self.shared_observation_spaces

    def _init_rendering(self):
        """
        Initialize the rendering of the environment.

        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        credit: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#rendering
        """
        self.clock = pygame.time.Clock()
        if self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Cops and Robbers")
            self.overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            if hasattr(self.map, "agent_spawn_regions"):
                for regions_list in self.map.agent_spawn_regions.values():
                    for region in regions_list:
                        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                        color = (255, 200, 100, 40)
                        pygame.draw.rect(self.overlay, color, pygame.Rect(x, y, w, h))
            if self.map_image:
                self._map_image = pygame.image.load(self.map_image)
                self._map_image = pygame.transform.scale(
                    self._map_image, (self.width, self.height)
                )
            self.font = pygame.font.Font(None, 36)
        elif self.render_mode == "rgb_array":
            self.window = pygame.Surface((self.width, self.height))

    def _get_info(self):
        """
        Get additional information about the current environment state.

        This method is meant to be overridden by subclasses to provide
        environment-specific debugging and monitoring information.

        Returns:
            dict: Information about the current environment state
        """
        # Base implementation with minimal information
        return {
            "step_count": self.step_count,
            "thief_count": len(self.thieves),
            "cop_count": len(self.cops),
            "terminated": any(self._termination_criterion()),
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        draw_options = pymunk.pygame_util.DrawOptions(self.window)

        if self.render_mode == "human":
            self.window.fill((255, 255, 255))
            self.window.blit(self.overlay, (0, 0))
            if self.map_image:
                self.window.blit(self._map_image, (0, 0))

            self.space.debug_draw(draw_options)

            fps = int(self.clock.get_fps())
            fps_text = self.font.render(f"FPS: {fps}", True, (0, 0, 0))
            self.window.blit(fps_text, (10, 10))
            step_text = self.font.render(f"Step: {self.step_count}", True, (0, 0, 0))
            self.window.blit(step_text, (10, 50))

            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit(0)
            self.clock.tick(60)

        elif self.render_mode == "rgb_array":
            pygame.event.get()
            return pygame.surfarray.array3d(self.window)

        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

    def observe(self, agent):
        return np.array(agent.get_observation(), dtype=np.float32)

    def _termination_criterion(self) -> tuple[bool, bool]:
        """
        Check if the termination condition has been met.

        The episode terminates when any thief is caught by any cop,
        which happens when they are within the termination radius of each other
        and no obstacles are between them.

        Returns:
            tuple [bool, bool]: A tuple containing two boolean values:
            - True if cops catch thieves (termination condition met)
            - True if the maximum step count is reached (truncation condition met)
        """
        for thief in self.thieves:
            for cop in self.cops:
                filter = pymunk.ShapeFilter(
                    mask=~(thief.filter_category | cop.filter_category) & 0xFFFFFFFF
                )
                hit = self.space.segment_query_first(
                    thief.body.position,
                    cop.body.position,
                    0.0,
                    shape_filter=filter,
                )
                if hit is None:
                    if (
                        thief.body.position.get_distance(cop.body.position)
                        < self._termination_radius
                    ):
                        return (True, False)
        if self.step_count >= self.max_step_count:
            return (False, True)

        return (False, False)


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
