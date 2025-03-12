import gymnasium as gym
import numpy as np


class Entity:
    """
    ## Entity Class

    Base class for all agents in the simulation.

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.

    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up

    ## Observation Space
    # TODO: Add observation space for the agent: 360 degree view with a possibility for narrower view.
    The observation space is dependent on the field of view of the agent.


    """

    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)  # 4 actions: left, down, right, up
        self.observation_space = None

    def step(self, action: np.ndarray):
        """
        Method to update the agent's state.
        # TODO: Add view of the agent
        """
        # return np.array(state, dtype=np.float32), reward, terminated, False, {}

        raise NotImplementedError
