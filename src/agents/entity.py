import gymnasium as gym
import numpy as np
import pymunk
import uuid

from utils import get_unit_size, get_unit_velocity, get_unit_mass


class Entity:
    """
    ## Entity Class

    Base class for all agents in the simulation.

    ## Action Space
    For now using a simplified action non-continuous space.
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.

    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up

    ## Observation Space
    # TODO: Add observation space for the agent: 360 degree view with a possibility for narrower view.
    The observation space is dependent on the field of view of the agent.


    """

    def __init__(
        self,
        radius: float | None = None,
        speed: float | None = None,
        mass: float | None = None,
        id: str | None = None,
        color: tuple[int, int, int] = (255, 255, 255),
    ):
        """
        Initialize the agent with the given radius and speed.
        """
        self._radius = get_unit_size() if radius is None else radius
        self._speed = get_unit_velocity() if speed is None else speed
        self._mass = get_unit_mass() if mass is None else mass
        # Generate a unique ID for the agent (needed for environment).
        self._id = uuid.uuid4().hex if id is None else id
        self.action_space = gym.spaces.Discrete(4)  # 4 actions: left, down, right, up
        # Force mappings for the actions of the agent.
        # 0: left, 1: down, 2: right, 3: up.
        self._force_mappings = {
            0: (-self._speed, 0),
            1: (0, -self._speed),
            2: (self._speed, 0),
            3: (0, self._speed),
        }

        self.observation_space = (
            None  # @TODO: Implement the observation space for the agent.
        )

        self.body = pymunk.Body(
            self._mass,
            pymunk.moment_for_circle(
                self._mass, inner_radius=0.0, outer_radius=self._radius
            ),
        )  # inner_radius = 0 because we represent filled circles as agents.

        self.radius = pymunk.Circle(self.body, radius=self._radius)

    def _perform_action(self, action: np.ndarray) -> None:
        """
        Performs an action in the environment.
        """
        # Map the action to a force direction:

        action_idx = int(action[0])
        force = self._force_mappings.get(action_idx, (0, 0))
        self.body.apply_force_at_local_point(force)

    def step(self, action: np.ndarray):
        """
        Method to update the agent's state.
        # TODO: Add view controller for the agent.
        """
        self._perform_action(action)
        raise NotImplementedError
        # return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def reset(self):
        raise NotImplementedError

    def get_observation(self):
        """
        Get the observation of the agent. For now it's its position.

        """
        return self.body.position

    def get_id(self) -> str:
        return self._id
