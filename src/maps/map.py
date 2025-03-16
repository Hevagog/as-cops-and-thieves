from typing import Tuple, Dict, List
import json
import os

import pymunk
import pymunk.pygame_util

import pygame


class Map:
    """
    Base class for the map in the environment.
    """

    def __init__(self, map_path: str) -> None:
        """
        Initialize the map.
        Args:
            map_path (str): Path to the JSON file containing the map.
        """
        self.window_dimensions: Tuple[int, int]  # width, height
        self.canvas_dimensions: Tuple[int, int]  # width, height
        self._scaling_factor_x: int  # scene can be larger/smaller than the window
        self._scaling_factor_y: int
        self.blocks = List[
            Dict[str, int]
        ]  # I have pytorch installed on python 3.10 so no type T_T

        self._parse_json_map(map_path)

    def _parse_json_map(self, map_path: str) -> None:
        """
        Generate the map from the given JSON file.
        Args:
            map_path (str): Path to the JSON file containing the map.
        """
        with open(map_path, "r") as f:
            map_data = json.load(f)
            self.window_dimensions = tuple(map_data["window"].values())
            self.canvas_dimensions = tuple(map_data["canvas"].values())
            self.blocks = map_data["objects"]["blocks"]

        self._scaling_factor_x = int(
            round(self.window_dimensions[0] / self.canvas_dimensions[0])
        )
        self._scaling_factor_y = int(
            round(self.window_dimensions[1] / self.canvas_dimensions[1])
        )

    def _physics_to_screen(self, x: int, y: int) -> Tuple[int, int]:
        """
        Convert physics coordinates (from the canvas) to screen coordinates.
        This is used ONLY for visualizing objects on the pygame display.

        Args:
            x (int): The x coordinate in the physics space.
            y (int): The y coordinate in the physics space.

        Returns:
            Tuple[int, int]: Rescaled screen coordinates.
        """
        return x * self._scaling_factor_x, y * self._scaling_factor_y

    def populate_space(self, space: pymunk.Space) -> None:
        """
        Populate the pymunk space with the map objects provided in the JSON map file.
        Args:
            space (pymunk.Space): The pymunk space to populate with the map objects.
        """
        for block in self.blocks:
            x, y = block.get("x"), block.get("y")
            w = block.get("w") if block.get("w") is not None else 1
            h = block.get("h") if block.get("h") is not None else 1

            block_segment = pymunk.Segment(space.static_body, (x, y), (x + w, y + h), 1)
            space.add(block_segment)

    def render(self, screen: pygame.Surface) -> None:
        """
        Visualize the map objects on the screen using the physics-to-screen coordinate conversion.
        This affects the display of the map on the pygame window, not the physics objects.
        Args:
            screen: The pygame surface to render the map on.
        """
        for block in self.blocks:
            x, y = block.get("x"), block.get("y")
            w = block.get("w") if block.get("w") is not None else 1
            h = block.get("h") if block.get("h") is not None else 1
            screen_x, screen_y = self._physics_to_screen(x, y)
            screen_w, screen_h = self._physics_to_screen(w, h)
            pygame.draw.rect(
                screen, (0, 0, 0), (screen_x, screen_y, screen_w, screen_h), 0
            )


### For now only for basic tests to showcase the map rendering
if __name__ == "__main__":
    map = Map("src\\maps\\maps_templates\\labyrinth.json")
    space = pymunk.Space()
    map.populate_space(space)

    pygame.init()
    screen = pygame.display.set_mode(map.window_dimensions)
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((255, 255, 255))
        map.render(screen)
        pygame.display.flip()
        clock.tick(60)
