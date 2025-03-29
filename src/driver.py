from environments import SimpleEnv
from maps import Map
from time import sleep
from pathlib import Path
import argparse


def configure_argparser():
    parser = argparse.ArgumentParser(description="Cops and Robbers Game")
    
    # Positional argument
    parser.add_argument(
        "mapfile",
        type=Path,
        help="The name of the map to process"
    )
    
    # Optional arguments
    parser.add_argument(
        "-k", 
        "--keep-alive",
        action="store_true",
        help="Keep the game window open after the game ends" 
    )
    parser.add_argument(
        "-i", 
        "--map-image",
        type=Path,
        default=None,
        help="Path to the map image"
    )
    parser.add_argument(
        "-r", 
        "--render-mode",
        type=str,
        choices=["human", "rgb_array"],
        default="human",
        help="Render mode for the environment (default: human)"
    )

    # parser.add_argument(
    #     "-n", 
    #     "--number", 
    #     type=int, 
    #     default=10, 
    #     help="Specify a number (default: 10)"
    # )
    
    return parser.parse_args()
    

def main():

    args = configure_argparser()

    map = Map(args.mapfile)
    env = SimpleEnv(
        map=map, 
        map_image=args.map_image,
        render_mode=args.render_mode
    )
    observations, infos = env.reset()
    for i in range(10_000):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, _, infos = env.step(actions)
        if i % 500 == 0:
            print(f"Step {i}")
            observations, infos = env.reset()

    if args.keep_alive:
        # @FIXME: this is a tmp solution to keep the window open
        # Keep the window open until the user presses Enter
        input("Press Enter to close the window...")

if __name__ == "__main__":
    main()
