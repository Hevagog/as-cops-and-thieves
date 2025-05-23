from environments import BaseEnv
from maps import Map
from pathlib import Path


class SimpleEnv(BaseEnv):
    """
    Simple environment implementation for the cops and thieves game.

    This environment extends BaseEnv with simplified initialization for
    common use cases, making it easier to create standard environments.
    """

    def __init__(
        self,
        map: Map,
        render_mode="rgb_array",
        map_image: Path = None,
        max_step_count: int = 400,
        time_step: float = 1 / 60.0,
    ):
        """
        Initialize a simplified environment for the cops and thieves game.

        Args:
            map: The map object for the environment
            render_mode: The mode in which to render the environment (either "human" or "rgb_array")
            map_image: Optional path to a map background image
            max_step_count: Maximum number of steps before truncation
            time_step: Physics simulation time step
        """
        super().__init__(
            map=map,
            map_image=map_image,
            render_mode=render_mode,
            max_step_count=max_step_count,
            time_step=time_step,
        )

    def _get_info(self):
        """
        Extend the base info method with environment-specific information.

        Returns:
            dict: Enhanced information about the current environment state
        """
        base_info = super()._get_info()
        # Add any SimpleEnv-specific information
        base_info.update(
            {
                "environment_type": "SimpleEnv",
                "thief_positions": [
                    thief.body.position.int_tuple for thief in self.thieves
                ],
                "cop_positions": [cop.body.position.int_tuple for cop in self.cops],
            }
        )
        return base_info
