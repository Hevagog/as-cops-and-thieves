import os
from contextlib import contextmanager
from typing import Any, Dict, Generator
import tomli


def get_unit_size() -> int:
    with _get_specific_config() as config:
        return config["unit_size"]


def get_unit_velocity() -> int:
    with _get_specific_config() as config:
        return config["unit_velocity"]


def get_unit_mass() -> int:
    with _get_specific_config() as config:
        return config["unit_mass"]


def get_window_width() -> int:
    with _get_specific_config(name="tool.visualization") as config:
        return config["window_shape"]


def get_window_height() -> int:
    with _get_specific_config(name="tool.visualization") as config:
        return config["window_shape"]


@contextmanager
def _get_specific_config(name: str = "tool.physical-params"):
    with open("config.toml") as f:
        config = tomli.load(f)
    yield config[name]
