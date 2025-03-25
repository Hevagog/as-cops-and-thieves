from environments import SimpleEnv
from maps import Map
from time import sleep


def main():
    map = Map("src\\maps\\maps_templates\\grandbyrinth.json")
    env = SimpleEnv(map=map, render_mode="human")
    observations, infos = env.reset()
    for i in range(10_000):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, _, infos = env.step(actions)
        if i % 500 == 0:
            print(f"Step {i}")
            observations, infos = env.reset()


if __name__ == "__main__":
    main()
