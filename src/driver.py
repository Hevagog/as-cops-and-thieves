from environments import SimpleEnv
from maps import Map


def main():
    map = Map("src\\maps\\maps_templates\\labyrinth.json")
    env = SimpleEnv(cops_count=1, thieves_count=1, map=map, render_mode="human")
    env.reset()


if __name__ == "__main__":
    main()
