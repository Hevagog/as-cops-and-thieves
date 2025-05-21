from agents.entity import Entity
import pymunk
import gymnasium as gym
import numpy as np

from utils import ObjectType


class Cop(Entity):
    """
    Cop agent; the agent that is trying to catch the thief.
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
        self.color = (0, 0, 255, 255)
        self.proximity_coeff = 0.05
        super().__init__(
            start_position=start_position,
            space=space,
            color=self.color,
            id=id,
            group=group,
            filter_category=filter_category,
        )

        self.observation_priorities = [
            ObjectType.THIEF,
            ObjectType.MOVABLE,
            ObjectType.COP,
            ObjectType.WALL,
            ObjectType.EMPTY,
        ]

    def reward(
        self, observation: gym.spaces.Dict, is_terminated: tuple[bool, bool]
    ) -> float:
        """
        Reward function for the cop agent.
        Args:
            observation (gym.spaces.Dict): The observation space of the agent:
                - distance : A numpy array containing the distances from the agent to the first object hit along each ray.
                - object_type : A numpy array containing the type of object detected for each ray, based on pre-defined ObjectType values.
            is_terminated (tuple[bool, bool]): A tuple indicating whether the game is terminated for the cop and thief agents.
        Returns:
            float: The reward for the cop agent.
        """
        if is_terminated[0]:
            return 1.0
        elif is_terminated[1]:
            return -1.0
        reward = 0.0
        thief_mask = observation["object_type"] == ObjectType.THIEF.value
        if thief_mask.any():
            d = observation["distance"][thief_mask].min()
            reward += 3.0 * np.exp(-d / 100.0) / 10.0  # Sharper reward for proximity
        else:
            reward -= 0.015  # Smaller penalty for not seeing
        # Encourage movement
        # velocity_norm = np.linalg.norm(self.body.velocity)
        # if velocity_norm < 5.0:
        #     reward -= 0.05
        return reward
