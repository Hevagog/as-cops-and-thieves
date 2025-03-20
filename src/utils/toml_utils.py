from contextlib import contextmanager
import tomli


def get_unit_size() -> int:
    with _get_specific_config() as config:
        return config["unit_size"]


def get_unit_velocity() -> int:
    with _get_specific_config() as config:
        return config["unit_velocity"]


def get_max_speed() -> int:
    with _get_specific_config() as config:
        return config["max_speed"]


def get_unit_mass() -> int:
    with _get_specific_config() as config:
        return config["unit_mass"]


def get_cop_start_category() -> int:
    with _get_specific_config() as config:
        return config["pymunk_cop_start_category"]


def get_thief_start_category() -> int:
    with _get_specific_config() as config:
        return config["pymunk_thief_start_category"]


@contextmanager
def _get_specific_config(keys=["tool", "physical-params"]):
    with open("pyproject.toml", "rb") as f:
        config = tomli.load(f)
    for key in keys:
        config = config[key]
    yield config
