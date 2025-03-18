from agents.entity import Entity


class Cop(Entity):
    """
    Cop agent; the agent that is trying to catch the thief.
    """

    def __init__(self, id=None):
        self.color = (0, 0, 255)
        super().__init__(color=self.color, id=id)
