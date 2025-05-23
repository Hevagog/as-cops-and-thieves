import random
import pymunk


def sample_spawn_position(region) -> pymunk.Vec2d:
    assert (
        "x" in region and "y" in region and "w" in region and "h" in region
    ), "Invalid spawn region format. Must contain x, y, w, h."
    x = random.uniform(region["x"], region["x"] + region["w"])
    y = random.uniform(region["y"], region["y"] + region["h"])
    return pymunk.Vec2d(x, y)
