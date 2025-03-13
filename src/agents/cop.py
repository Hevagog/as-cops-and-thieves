from agents.entity import Entity


class Cop(Entity):
    """
    Cop agent; the agent that is trying to catch the thief.
    """

    def __init__(self):
        super().__init__()
        self.color = (0, 0, 255)
