from typing import Tuple, Dict, List
import json
import os
import pymunk
import pymunk.pygame_util
import pygame

from utils import get_unit_size


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
        self.cops_count: int
        self.thieves_count: int
        self.cops_positions: List[Tuple[int, int]]
        self.thieves_positions: List[Tuple[int, int]]
        self.unit_size = get_unit_size()

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
            self.blocks = map_data["objects"]["blocks"]
            agents = map_data["agents"]

            self.cops_count = sum(1 for agent in agents if agent["type"] == "cop")
            self.thieves_count = sum(1 for agent in agents if agent["type"] == "thief")
            self.cops_positions = [
                (agent["x"], agent["y"]) for agent in agents if agent["type"] == "cop"
            ]
            self.thieves_positions = [
                (agent["x"], agent["y"]) for agent in agents if agent["type"] == "thief"
            ]

    def populate_space(self, space: pymunk.Space) -> None:
        """
        Populate the pymunk space with the map objects provided in the JSON map file.
        Args:
            space (pymunk.Space): The pymunk space to populate with the map objects.
        """

        for block in self.blocks:
            x, y = block.get("x"), block.get("y")
            x2, y2 = block.get("x2"), block.get("y2")
            block_segment = pymunk.Segment(
                space.static_body, (x, y), (x2, y2), self.unit_size
            )
            space.add(block_segment)


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
        map.render(screen, [])
        pygame.display.flip()
        clock.tick(60)
