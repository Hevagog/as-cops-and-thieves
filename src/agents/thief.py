from agents.entity import Entity


class Thief(Entity):
    """
    Class for the thief agent; i.e. the agent that is trying to escape from the cops.
    """

    def __init__(self):
        super().__init__()
        self.color = (255, 0, 0)
