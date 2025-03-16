from environments import BaseEnv
from maps import Map


class SimpleEnv(BaseEnv):
    def __init__(
        self, cops_count: int, thieves_count: int, map: Map, render_mode="rgb_array"
    ):
        """
        Simple environment for the cops and thieves game.
        # TODO: Is it feasible to have base_env.py and simple_env.py? Maybe we can have more generalized BaseEnv
        Args:
            cops_count: Number of cops in the environment
            thieves_count: Number of thieves in the environment
            map: The map object for the environment
            render_mode: The mode in which to render the environment (either "human" or "rgb_array")
        """
        super().__init__(
            cops_count=cops_count,
            thieves_count=thieves_count,
            map=map,
            render_mode=render_mode,
        )
