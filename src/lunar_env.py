import argparse

import gymnasium as gym

from dqn import DQN


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN on LunarLander.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[16, 16])
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--target-update-frequency", type=int, default=100)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch_size > args.replay_capacity:
        raise ValueError("batch-size cannot be larger than replay-capacity")

    render_mode = "human" if args.render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    dqn = DQN(
        hidden_layers=args.hidden_layers,
        target_update_frequency=args.target_update_frequency,
        gamma=args.gamma,
        replay_capacity=args.replay_capacity,
        learning_rate=args.learning_rate,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
    )

    state, info = env.reset()

    for _ in range(args.steps):
        action = dqn.choose_action(state)
        current_state = state
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        dqn.update_experience_replay(
            current_state,
            action,
            reward,
            next_state,
            done,
        )
        dqn.replay(args.batch_size)
        state = next_state

        if done:
            state, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
