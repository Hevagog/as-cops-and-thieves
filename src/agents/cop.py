from agents.entity import Entity
import pymunk


class Cop(Entity):
    """
    Cop agent; the agent that is trying to catch the thief.
    """

    def __init__(
        self,
        start_position: pymunk.Vec2d,
        space: pymunk.Space,
        group: int,
        id=None,
        filter_category: int = 0b1,
    ):
        self.color = (0, 0, 255, 255)
        super().__init__(
            start_position=start_position,
            space=space,
            color=self.color,
            id=id,
            group=group,
            filter_category=filter_category,
        )
