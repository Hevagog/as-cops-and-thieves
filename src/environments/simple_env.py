from environments import BaseEnv


class SimpleEnv(BaseEnv):
    def __init__(self, cops_count: int, thieves_count: int):
        """
        Simple environment for the cops and thieves game.
        Args:
            cops_count: Number of cops in the environment
            thieves_count: Number of thieves in the environment
        """
        super().__init__(cops_count=cops_count, thieves_count=thieves_count)

    def step(self, action):
        # Implement the step method
        pass

    def reset(self):
        # Implement the reset method
        pass

    def render(self):
        # Implement the render method
        pass

    def close(self):
        # Implement the close method
        pass
