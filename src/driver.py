from environments import SimpleEnv
from maps import Map
from time import sleep
from pathlib import Path
import argparse
import pygame as pg


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
    
    # initial reset
    observations, infos = env.reset()

    def step_machen_dawaj_burwo_step():
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, _, infos = env.step(actions)
        # the above should be handeled here

    auto_step = False
    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

            if event.type == pg.KEYDOWN:
                if event.key in [pg.K_ESCAPE, pg.K_q]:
                    running = False
                if event.key == pg.K_SPACE:
                    auto_step = not auto_step
                if event.key == pg.K_RIGHT:
                    auto_step = False
                    step_machen_dawaj_burwo_step()
                                       
        if auto_step:
            step_machen_dawaj_burwo_step()
            
    env.close()

if __name__ == "__main__":
    main()
