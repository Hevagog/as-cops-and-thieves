from agents.entity import Entity
import pymunk


class Thief(Entity):
    """
    Class for the thief agent; i.e. the agent that is trying to escape from the cops.
    """

    def __init__(self, start_position: pymunk.Vec2d, space: pymunk.Space, id=None):
        self.color = (255, 0, 0)
        super().__init__(
            start_position=start_position, space=space, color=self.color, id=id
        )
