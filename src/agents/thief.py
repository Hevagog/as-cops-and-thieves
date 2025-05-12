from agents.entity import Entity
import pymunk
import gymnasium as gym
import numpy as np

from utils import ObjectType


class Thief(Entity):
    """
    Class for the thief agent; i.e. the agent that is trying to escape from the cops.
    """

    def __init__(
        self,
        start_position: pymunk.Vec2d,
        space: pymunk.Space,
        group: int,
        filter_category: int = 0b1,
        id: str | None = None,
    ):
        """
        Args:
            start_position (pymunk.Vec2d): The starting position of the agent.
            space (pymunk.Space): The pymunk space in which the agent will be placed.
            group (int): The group to which the agent belongs. Every agent has a distinct group value
            filter_category(int): The category for the agent. It is more general than group. It is used to distinguish between different types of agents.
            id (str | None): The id of the agent.
        """
        self.color = (255, 0, 0, 255)
        super().__init__(
            start_position=start_position,
            space=space,
            color=self.color,
            id=id,
            group=group,
            filter_category=filter_category,
        )

        self.observation_priorities = [
            ObjectType.COP,
            ObjectType.MOVABLE,
            ObjectType.THIEF,
            ObjectType.WALL,
            ObjectType.EMPTY,
        ]

    def reward(self, observation: gym.spaces.Dict, is_terminated: bool) -> float:
        """
        Reward function for the thief agent.
        Args:
            observation (gym.spaces.Dict): The observation space of the agent:
                - distance : A numpy array containing the distances from the agent to the first object hit along each ray.
                - object_type : A numpy array containing the type of object detected for each ray, based on pre-defined ObjectType values.
            is_terminated (tuple[bool, bool]): A tuple indicating whether the game is terminated for the cop and thief agents.
        Returns:
            float: The reward for the thief agent.
        """
        if is_terminated[0]:
            return -10.0
        if is_terminated[1]:
            return 10.0
        cop_mask = observation["object_type"] == ObjectType.COP.value
        if cop_mask.any():
            min_cop_distance = np.min(observation["distance"][cop_mask])
            return np.tanh((min_cop_distance - 100.0) / 50.0)
            # return np.exp((min_cop_distance - 200.0) / 200.0) - 2.0
        else:
            return 0.5

        # TODO: Check validity of this reward scheme:
        #  Small penalty for low velocity to discourage just standing still
        # velocity_norm = np.linalg.norm(self.body.velocity)
        # if velocity_norm < 5.0: # some threshold
        #     reward -= 0.05
