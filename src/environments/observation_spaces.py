import gymnasium as gym
import numpy as np
from typing import List
import functools


from maps import Map
from utils import ObjectType
from agents.cop import Cop
from agents.thief import Thief


def _init_shared_observation_space(
    map: Map, cops: List[Cop], thieves: List[Thief]
) -> gym.spaces.Dict:
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
    max_dim = max(map.window_dimensions)

    # Process both teams: cops and thieves.
    for team_agents in [cops, thieves]:
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


def get_shared_observations(
    observations, cops: List[Cop], thieves: List[Thief]
) -> dict:
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
    cop_positions = np.array([cop.body.position for cop in cops], dtype=np.float16)
    thief_positions = np.array(
        [thief.body.position for thief in thieves], dtype=np.float16
    )

    for team_name, team_agents, team_positions in [
        ("cops", cops, cop_positions),
        ("thieves", thieves, thief_positions),
    ]:
        if not team_agents:
            continue

        first_agent = team_agents[0]
        obj_masked = (
            np.ones_like(observations[first_agent.get_id()]["object_type"])
            * ObjectType.EMPTY.value
        )
        dist_masked = np.zeros_like(observations[first_agent.get_id()]["distance"])

        for agent in team_agents:
            obs = observations[agent.get_id()]
            obj_types = obs["object_type"]
            distances = obs["distance"]

            for priority_type in agent.observation_priorities:
                mask = (obj_types == priority_type.value) & (
                    obj_masked == ObjectType.EMPTY.value
                )
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


@functools.lru_cache(maxsize=None)
def get_nested_agent_observation_spaces(
    shared_observation_spaces: gym.spaces.Dict,
) -> gym.spaces:
    """
    Get a flattened version of agent observation spaces suitable for MAPPO.

    Instead of deep nesting, this creates a flat structure where each agent's
    space includes their own observations plus flattened versions of others'.

    Returns:
        gym.spaces.Dict: Dictionary of flattened observation spaces
    """
    flat_spaces = {}

    for agent_id in shared_observation_spaces:
        # Start with agent's own space
        agent_space = {}
        for key, space in shared_observation_spaces[agent_id].spaces.items():
            agent_space[key] = space

        # Add flattened versions of other agents' spaces
        for other_id in shared_observation_spaces:
            if other_id != agent_id:
                for key, space in shared_observation_spaces[other_id].spaces.items():
                    agent_space[f"{other_id}_{key}"] = space

        flat_spaces[agent_id] = gym.spaces.Dict(agent_space)

    return gym.spaces.Dict(flat_spaces)
