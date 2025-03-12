from environments import BaseEnv


class SimpleEnv(BaseEnv):
    def __init__(self):
        super().__init__()

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
