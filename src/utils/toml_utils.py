import os
from contextlib import contextmanager
from typing import Any, Dict, Generator
import tomli


def get_unit_size() -> int:
    with _get_cm_config() as config:
        return config["unit_size"]


def get_velocity() -> int:
    with _get_cm_config() as config:
        return config["velocity"]


@contextmanager
def _get_cm_config():
    with open("config.toml") as f:
        config = tomli.load(f)
    yield config["tool.tbd"]
