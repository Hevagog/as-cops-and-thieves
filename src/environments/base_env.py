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

    def __init__(self, map: Map, map_image: Path = None, render_mode=None):
        """
        Initialize the cops and thieves environment with map configuration and rendering options.

        Args:
            map (Map): The map object defining the environment layout, obstacles, and agent positions.
            map_image (Path, optional): Path to an image file for background rendering.
            render_mode (str, optional): Visualization mode, either "human" for interactive display
                                        or "rgb_array" for programmatic access.
        """
        super().__init__()

        # Define Space
        self.map = map
        self.map_image = map_image
        self.width, self.height = self.map.window_dimensions
        self.space = pymunk.Space()
        self.map.populate_space(self.space)

        self.step_count = 0

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
        shared_obs_space = self._init_shared_observation_space()
        self.shared_observation_spaces = shared_obs_space
        self._shared_observation_spaces = shared_obs_space
        self.state_space = shared_obs_space

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._init_rendering()

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

    def _init_shared_observation_space(self) -> gym.spaces.Dict:
        """
        Initialize the structured observation space for shared team information.

        Creates a hierarchical dictionary of gym spaces where:
        1. The top level contains agent IDs as keys
        2. Each agent's space contains five components:
           - own_obj_types: The agent's original object type observations
           - own_distances: The agent's original distance observations
           - object_type_shared: Team-aggregated object type observations
           - distance_shared: Team-aggregated distance observations
           - team_positions: 2D coordinates of all team members

        Returns:
            gym.spaces.Dict: Nested dictionary of observation spaces by agent ID
        """
        shared_spaces = {}
        max_dim = max(self.map.window_dimensions)

        # Process both teams: cops and thieves.
        for team_agents in [self.cops, self.thieves]:
            if not team_agents:
                continue

            # Use the first agent of the team as an example
            example_obs_space = team_agents[0].observation_space
            obj_space = example_obs_space["object_type"]
            dist_space = example_obs_space["distance"]

            team_positions_space = gym.spaces.Box(
                low=0.0,
                high=max_dim,
                shape=(len(team_agents), 2),
                dtype=np.float16,
            )

            team_shared_space = gym.spaces.Dict(
                {
                    "own_obj_types": obj_space,
                    "own_distances": dist_space,
                    "object_type_shared": obj_space,
                    "distance_shared": dist_space,
                    "team_positions": team_positions_space,
                }
            )

            for agent in team_agents:
                shared_spaces[agent.get_id()] = team_shared_space

        return gym.spaces.Dict(shared_spaces)

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
        Get a hierarchically nested version of all agent observation spaces.

        Creates a nested structure where each agent's space contains
        references to all other agents' spaces. This allows agents to
        potentially access information from other agents during training.

        This is needed for MAPPO training.

        Returns:
            gym.spaces.Dict: Nested dictionary of observation spaces
        """
        shared_spaces = {
            agent_id: gym.spaces.Dict(
                {
                    # Copy the agent's own space first
                    **self._shared_observation_spaces[agent_id],
                    # Add other agents' spaces as nested entries
                    **{
                        other_id: gym.spaces.Dict(other_space)
                        for other_id, other_space in self._shared_observation_spaces.items()
                        if other_id != agent_id
                    },
                }
            )
            for agent_id in self._shared_observation_spaces
        }

        return gym.spaces.Dict(shared_spaces)

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

        self.agents = self.possible_agents[:]
        for agent in self.agents:
            self.agent_name_mapping[agent].reset()

        observations = {
            agent: self.agent_name_mapping[agent].get_observation()
            for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}

        self.shared_observation_spaces = self.get_shared_observations(observations)

        if self.render_mode == "human":
            self._render_frame()

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
        observations, rewards, terminations, truncations, infos = [
            dict(zip(self.agents, values)) for values in zip(*results)
        ]

        self.shared_observation_spaces = self.get_shared_observations(observations)

        self.space.step(1 / 60.0)  # @TODO: Parameterize the time step

        if self.render_mode == "human":
            self._render_frame()

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
        if self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Cops and Robbers")
            self.clock = pygame.time.Clock()
        elif self.render_mode == "rgb_array":
            self.window = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()

    def _get_info(self):
        raise NotImplementedError

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        draw_options = pymunk.pygame_util.DrawOptions(self.window)

        if self.render_mode == "human":
            self.window.fill((255, 255, 255))

            if self.map_image:
                map_image = pygame.image.load(self.map_image)
                map_image.set_alpha(
                    int(0.75 * 255)
                )  # Set opacity (0 is fully transparent, 255 is fully opaque)
                map_image = pygame.transform.scale(map_image, (self.width, self.height))
                self.window.blit(map_image, (0, 0))

            self.space.debug_draw(draw_options)

            font = pygame.font.Font(None, 36)
            fps = int(self.clock.get_fps())
            fps_text = font.render(f"FPS: {fps}", True, (0, 0, 0))
            self.window.blit(fps_text, (10, 10))
            step_text = font.render(f"Step: {self.step_count}", True, (0, 0, 0))
            self.window.blit(step_text, (10, 50))

            pygame.display.flip()
            self.clock.tick(60)

        elif self.render_mode == "rgb_array":
            self.space.debug_draw(draw_options)
            return pygame.surfarray.array3d(self.window)

        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")

        self.clock.tick(60)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

    def observe(self, agent):
        return np.array(agent.get_observation(), dtype=np.float32)

    def _termination_criterion(self) -> bool:
        """
        Check if the termination condition has been met.

        The episode terminates when any thief is caught by any cop,
        which happens when they are within the termination radius of each other
        and no obstacles are between them.

        Returns:
            bool: True if the termination condition is met, False otherwise
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
        Create team-based shared observations with appropriate information masking.

        This method aggregates observations across team members according to priority rules:
        1. Each agent contributes information about object types they can observe
        2. Higher-priority information overwrites lower-priority information
        3. Team positions are shared completely among teammates

        The resulting observations preserve individual perception while adding team knowledge.

        Args:
            observations (dict): Individual agent observations keyed by agent ID

        Returns:
            dict: Enhanced observations with team-shared information, containing:
                - own_obj_types: Original individual object type observations
                - own_distances: Original individual distance measurements
                - object_type_shared: Team-aggregated object type information
                - distance_shared: Team-aggregated distance information
                - team_positions: Positions of all team members
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

                shared_observations[agent.get_id()] = {
                    "own_obj_types": obj_types,
                    "own_distances": distances,
                    "object_type_shared": obj_masked,
                    "distance_shared": dist_masked,
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
