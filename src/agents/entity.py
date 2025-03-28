import gymnasium as gym
import numpy as np
import pygame
import pymunk
import uuid
import math

from utils import (
    get_unit_size,
    get_unit_velocity,
    get_unit_mass,
    get_max_speed,
    get_thief_category,
    ObjectType,
)


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
    The observation space is a dictionary with two keys:
    - `distance`: A numpy array containing the distances from the agent to the first object hit along each ray.
    - `object_type`: A numpy array containing the type of object detected for each ray, based on pre-defined ObjectType values.

    """

    def __init__(
        self,
        start_position: pymunk.Vec2d,
        space: pymunk.Space,
        group: int,
        radius: float | None = None,
        speed: float | None = None,
        mass: float | None = None,
        id: str | None = None,
        filter_category: int = 0b1,
        color: tuple[int, int, int, int] = (0, 0, 0, 255),
    ):
        """
        Initialize the agent.

        Args:
            radius (float): The radius of the agent.
            speed: The speed of the agent.
            filter_category: The category for agent.
            mass: The mass of the agent.
            id: The unique ID of the agent.
            group: unique group for each agent for collision filtering.
            color: The color of the agent in RGBA format.
        """
        self._radius = get_unit_size() if radius is None else radius
        self._speed = get_unit_velocity() if speed is None else speed
        self._mass = get_unit_mass() if mass is None else mass
        self._id = uuid.uuid4().hex if id is None else id
        self._max_speed = get_max_speed()
        self.action_space = gym.spaces.Discrete(4)
        self._space = space
        self._thief_category = get_thief_category()
        self._initial_position = start_position

        # Define force mappings for actions:
        # 0: left, 1: down, 2: right, 3: up.
        self._force_mappings = {
            0: pymunk.Vec2d(-self._speed, 0),
            1: pymunk.Vec2d(0, self._speed),
            2: pymunk.Vec2d(self._speed, 0),
            3: pymunk.Vec2d(0, -self._speed),
        }

        self._ray_length = 400  # Length of the ray for the agent's view.
        self._fov = 2 * np.pi  # Field of view in radians.
        self._num_rays = 90  # One ray every ~4 degrees.

        # Observation space includes distance and object type for each ray.
        self.observation_space = gym.spaces.Dict(
            {
                "distance": gym.spaces.Box(
                    low=0.0,
                    high=self._ray_length,
                    shape=(self._num_rays,),
                    dtype=np.float16,
                ),
                "object_type": gym.spaces.MultiDiscrete(
                    [max(item.value for item in ObjectType)] * self._num_rays
                ),
            }
        )

        # Create the physics body and bounding circle representing the agent.
        self.body = pymunk.Body(
            self._mass,
            pymunk.moment_for_circle(
                self._mass, inner_radius=0.0, outer_radius=self._radius
            ),
        )
        self.body.position = self._initial_position
        self._b_box = pymunk.Circle(self.body, radius=self._radius)
        self._b_box.color = color
        self._b_box.filter = pymunk.ShapeFilter(group=group, categories=filter_category)

        self.ray_filter = pymunk.ShapeFilter(
            group=group,
            categories=filter_category,
        )
        self._space.add(self.body, self._b_box)

    def _perform_action(self, action: int) -> None:
        """
        Performs an action in the environment.
        """
        # Map the action to a force direction:
        force = self._force_mappings.get(action)
        self.body.apply_impulse_at_local_point(force)
        if abs(self.body.velocity) > self._max_speed:
            self.body.velocity = self.body.velocity.normalized() * self._max_speed

    def step(self, action: int):
        """
        Performs a step in the environment based on the action taken.
        Args:
            action: The action to be performed.
        """
        self._perform_action(action)
        observations = self.get_observation()
        # TODO: Add reward calculation based on the agent's state and actions.
        reward = 0
        terminated = False

        return observations, reward, terminated, False, {}

    def reset(self):
        """
        Resets the agent to its initial state.
        """
        self.body.position = self._initial_position
        self.body.velocity = pymunk.Vec2d(0, 0)

    def get_observation(self) -> dict[str, np.ndarray]:
        """
        Computes the observation for the agent by performing multiple ray casts within its field of view.

        This method projects an array of rays from the agent's current position, equally distributed across a specified field of view (FOV). For each ray, it computes an endpoint based on the agent's position, the ray length, and the ray's angle. A segment query is then performed to determine if the ray collides with any object in the environment.

        For each ray:
        - If a collision is detected (hit):
            - The distance from the agent to the point of collision is calculated.
            - The type of object hit is determined:
                - If the hit body is dynamic:
                    - If the shape is a polygon, it is classified as a movable object.
                    - If the shape is a circle, it is classified either as a thief or a cop, based on its filter categories.
                    - Otherwise, it defaults to an empty classification.
                - If the body is not dynamic, it is classified as a wall.
        - If no collision occurs:
            - The default ray length is recorded as the distance.
            - The object type is marked as empty.
        Returns:
            A dictionary with the following keys:
                "distance": A numpy array (dtype=np.float16) containing the distances from the agent to the first object hit along each ray.
                "object_type": A numpy array (dtype=np.uint8) containing the type of object detected for each ray, based on pre-defined ObjectType values.
        """
        angles = np.linspace(0, self._fov, self._num_rays, endpoint=False)
        cosines = np.cos(angles)
        sines = np.sin(angles)

        origin = self.body.position
        origin_x, origin_y = origin[0], origin[1]
        ray_length = self._ray_length
        ray_filter = self.ray_filter

        endpoints = np.column_stack(
            (origin_x + ray_length * cosines, origin_y + ray_length * sines)
        )

        hits = [
            self._space.segment_query_first(origin, pymunk.Vec2d(*end), 1, ray_filter)
            for end in endpoints
        ]

        distances = np.full(self._num_rays, ray_length, dtype=np.float16)
        object_types = np.full(self._num_rays, ObjectType.EMPTY.value, dtype=np.uint8)

        valid_idx = [i for i, hit in enumerate(hits) if hit is not None]
        if valid_idx:
            valid_hits = [hits[i] for i in valid_idx]
            valid_points = np.array([hit.point for hit in valid_hits], dtype=np.float16)

            dx = valid_points[:, 0] - origin_x
            dy = valid_points[:, 1] - origin_y
            distances[valid_idx] = np.hypot(dx, dy).astype(np.float16)

            object_types[valid_idx] = np.array(
                [self._query_body(hit.shape) for hit in valid_hits],
                dtype=np.uint8,
            )

        return {
            "distance": distances,
            "object_type": object_types,
        }

    def _query_body(self, hit_shape: pymunk.Shape) -> ObjectType:
        """
        Determines the type of object hit by the ray based on its properties.
        Args:
            hit_shape (pymunk.Shape): The shape that was hit by the ray.
        Returns:
            ObjectType: The type of object hit by the ray.
        """
        if hit_shape.body.body_type == pymunk.Body.DYNAMIC:
            if isinstance(hit_shape, pymunk.Poly):
                return ObjectType.MOVABLE.value
            elif isinstance(hit_shape, pymunk.Circle):
                if hit_shape.filter.categories == self._thief_category:
                    return ObjectType.THIEF.value
                else:
                    return ObjectType.COP.value
            else:
                return ObjectType.EMPTY.value
        else:
            return ObjectType.WALL.value

    def get_id(self) -> str:
        return self._id

    def reward(
        self, observation: gym.spaces.Dict, capture_condition: callable
    ) -> float:
        raise NotImplementedError("This method ought to be implemented in subclasses.")
