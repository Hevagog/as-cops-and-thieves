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

    def reward(self, observation: gym.spaces.Dict, is_terminated: bool) -> float:
        """
        Reward function for the cop agent.


        Returns:
            float: The reward for the cop agent.
        """
        # If thief gets caught, give a penalty of -10.0. Episode terminates.
        if is_terminated:
            reward = 10.0
        else:
            # Survival reward
            reward = 0.1
            # Calculate safety: reward increases with distance from cops
            cop_indices = [
                i
                for i, obj in enumerate(observation["object_type"])
                if obj == ObjectType.COP.value
            ]
            if cop_indices:
                d = np.min([observation["distance"][i] for i in cop_indices])
                lambda_thief = 0.05
                reward += lambda_thief * (d / observation["distance"].max())
            else:
                # If if no cops are detected, give a reward for being hidden
                reward += 0.5

        return reward
