from environments import SimpleEnv


def main():
    env = SimpleEnv(cops_count=1, thieves_count=1)
    env.reset()


if __name__ == "__main__":
    main()
