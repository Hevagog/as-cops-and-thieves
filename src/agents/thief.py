from agents.entity import Entity
import pymunk


class Thief(Entity):
    """
    Class for the thief agent; i.e. the agent that is trying to escape from the cops.
    """

    def __init__(
        self,
        start_position: pymunk.Vec2d,
        space: pymunk.Space,
        group: int,
        id=None,
        filter_category: int = 0b1,
    ):
        self.color = (255, 0, 0, 255)
        super().__init__(
            start_position=start_position,
            space=space,
            color=self.color,
            id=id,
            group=group,
            filter_category=filter_category,
        )
