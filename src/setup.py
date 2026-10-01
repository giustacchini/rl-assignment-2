import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN on LunarLander.")

    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[16, 16])
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--target-update-frequency", type=int, default=100)

    parser.add_argument("--checkpoint-frequency", type=int, default=1000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--render", action="store_true")

    return parser.parse_args()
