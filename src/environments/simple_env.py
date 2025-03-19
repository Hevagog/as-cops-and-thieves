from environments import BaseEnv
from maps import Map


class SimpleEnv(BaseEnv):
    def __init__(self, map: Map, render_mode="rgb_array"):
        """
        Simple environment for the cops and thieves game.
        # TODO: Is it feasible to have base_env.py and simple_env.py? Maybe we can have more generalized BaseEnv
        Args:
            map: The map object for the environment
            render_mode: The mode in which to render the environment (either "human" or "rgb_array")
        """
        super().__init__(
            map=map,
            render_mode=render_mode,
        )
