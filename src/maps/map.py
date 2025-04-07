from typing import Tuple, List
import json
import shapely as shp
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
        self.blocks = List[shp.Polygon]
        self.cops_count: int
        self.thieves_count: int
        self.cops_positions: List[Tuple[int, int]]
        self.thieves_positions: List[Tuple[int, int]]
        self.unit_size = get_unit_size()

        self._parse_json_map(map_path)

    def _parse_block(self, blk_json):
        blk_type = blk_json.get("type", "rect")  # default to rect
        if blk_type == "rect":
            x, y = blk_json.get("x"), blk_json.get("y")
            if x is None or y is None:
                raise ValueError(
                    "x and y coordinates are required for rectangle blocks."
                )
            # w and h are optional, default to 1
            w = blk_json.get("w") if blk_json.get("w") is not None else 1
            h = blk_json.get("h") if blk_json.get("h") is not None else 1
            vs = [
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h),
                (x, y),
            ]
        elif blk_type == "poly":
            vs = blk_json.get("vs")
            if vs is None:
                raise ValueError("Vertices are required for polygon blocks.")
            # Convert vertices to tuples
            vs = [(v.get("x"), v.get("y")) for v in vs]
        else:
            raise ValueError(f"Unknown block type: {blk_type}")
        return shp.Polygon(vs)

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
            blocks = map_data["objects"]["blocks"]
            self.blocks = [self._parse_block(block) for block in blocks]

            agents = map_data["agents"]
            self.cops_positions = [
                (agent["x"], agent["y"]) for agent in agents if agent["type"] == "cop"
            ]
            self.thieves_positions = [
                (agent["x"], agent["y"]) for agent in agents if agent["type"] == "thief"
            ]
            self.cops_count = len(self.cops_positions)
            self.thieves_count = len(self.thieves_positions)

    def populate_space(self, space: pymunk.Space) -> None:
        """
        Populate the pymunk space with the map objects provided in the JSON map file.
        Args:
            space (pymunk.Space): The pymunk space to populate with the map objects.
        """
        for block in self.blocks:
            vs = list(block.exterior.coords)
            block_segment = pymunk.Poly(space.static_body, vs, radius=1)
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
