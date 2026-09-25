# python
# lunar_env.py


import os
import json
import argparse
import gymnasium as gym
from datetime import datetime

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

    episode_returns = []
    episode_counter = 1

    episode_return = 0

    for _ in range(args.steps):
        action = dqn.choose_action(state)
        current_state = state
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_return += reward

        dqn.update_experience_replay(
            current_state,
            action,
            reward,
            next_state,
            done,
        )
        if len(dqn.experience_replay) >= args.batch_size:
            states, actions, rewards, next_states, dones = dqn.sample_experiences(batch_size=args.batch_size)
            dqn.train(states, actions, rewards, next_states, dones)
        state = next_state



        if done:
            episode_returns.append(episode_return)
            average_return = sum(episode_returns[-100:]) / len(episode_returns[-100:])
            print(
                f"Episode {episode_counter}: return = {episode_return}, average = {average_return}, epsilon = {dqn.epsilon}"
            )
            episode_counter += 1
            episode_return = 0
            state, info = env.reset()

    env.close()

    # save model and metric here using model.save()  
    project_location = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(f"results/{project_location}", exist_ok=True)
    dqn.network.save(f"results/{project_location}/training_model.keras")

    metrics = {
        "configuration": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "hidden_layers": args.hidden_layers,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "epsilon_decay": args.epsilon_decay,
            "epsilon_min": args.epsilon_min,
            "replay_capacity": args.replay_capacity,
            "target_update_frequency": args.target_update_frequency
        },
        "episode_returns": episode_returns,
        "final_epsilon": dqn.epsilon
    }
    with open(f"results/{project_location}/metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
