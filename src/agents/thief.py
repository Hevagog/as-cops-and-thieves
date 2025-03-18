from agents.entity import Entity


class Thief(Entity):
    """
    Class for the thief agent; i.e. the agent that is trying to escape from the cops.
    """

    def __init__(self, id=None):
        self.color = (255, 0, 0)
        super().__init__(color=self.color, id=id)
