# python
# lunar_env_episode_epsilon.py

import argparse
import json
import math
import os
import pickle as pkl
from datetime import datetime

import gymnasium as gym
import keras

from dqn import DQN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DQN with epsilon decay per episode."
    )
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
    parser.add_argument("--checkpoint-frequency", type=int, default=10_000)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main():
    project_location = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"results/{project_location}"
    os.makedirs(result_path, exist_ok=True)

    args = parse_args()
    if args.batch_size > args.replay_capacity:
        raise ValueError("batch-size cannot be larger than replay-capacity")

    configuration = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "hidden_layers": args.hidden_layers,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "epsilon_decay": args.epsilon_decay,
        "epsilon_min": args.epsilon_min,
        "epsilon_decay_frequency": "episode",
        "replay_capacity": args.replay_capacity,
        "target_update_frequency": args.target_update_frequency,
    }

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

    # DQN.train() normally decays epsilon per training update. Disable that
    # behavior here because this script decays epsilon once per episode below.
    dqn.epsilon_decay = 1.0

    initial_step = 1
    state, info = env.reset()

    episode_returns = []
    episode_steps = []
    episode_fuel_consumptions = []
    episode_successes = []
    episode_landing_errors = []
    episode_metrics = []

    episode_counter = 1
    episode_return = 0
    current_episode_steps = 0
    episode_fuel_consumption = 0

    if args.resume is not None:
        with open(f"{args.resume}/training_state.pkl", "rb") as file:
            loaded = pkl.load(file)

        initial_step = loaded["step"] + 1
        dqn.network = keras.saving.load_model(
            f"{args.resume}/training_model.keras"
        )
        dqn.target_network = keras.saving.load_model(
            f"{args.resume}/target_network.keras"
        )
        dqn.epsilon = loaded["epsilon"]
        dqn.experience_replay = loaded["experience_replay"]
        dqn.train_steps = loaded["train_steps"]
        episode_returns = loaded["episode_returns"]
        episode_steps = loaded.get("episode_steps", [])
        current_episode_steps = loaded.get("current_episode_steps", 0)
        episode_counter = loaded["episode_counter"]
        episode_fuel_consumptions = loaded["episode_fuel_consumptions"]
        episode_successes = loaded["episode_successes"]
        episode_landing_errors = loaded["episode_landing_errors"]
        episode_metrics = loaded.get("episode_metrics", [])

    episode_metrics_path = f"{result_path}/episode_metrics.jsonl"
    with open(episode_metrics_path, "w") as metrics_file:
        for metric in episode_metrics:
            metrics_file.write(json.dumps(metric) + "\n")

    for step in range(initial_step, args.steps + 1):
        action = dqn.choose_action(state)
        current_state = state
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_done = terminated or truncated
        current_episode_steps += 1

        if action == 2:
            episode_fuel_consumption += 0.3
        elif action in (1, 3):
            episode_fuel_consumption += 0.03

        episode_return += reward
        dqn.update_experience_replay(
            current_state,
            action,
            reward,
            next_state,
            terminated,
        )

        if len(dqn.experience_replay) >= args.batch_size:
            states, actions, rewards, next_states, terminateds = (
                dqn.sample_experiences(args.batch_size)
            )
            dqn.train(states, actions, rewards, next_states, terminateds)

        state = next_state

        if episode_done:
            success = bool(reward == 100)
            landing_error = float(abs(next_state[0])) if success else None
            landing_speed = math.hypot(next_state[2], next_state[3])

            # This is the only epsilon update in this script.
            dqn.epsilon = max(
                args.epsilon_min,
                dqn.epsilon * args.epsilon_decay,
            )

            episode_returns.append(episode_return)
            episode_steps.append(current_episode_steps)
            episode_fuel_consumptions.append(episode_fuel_consumption)
            episode_successes.append(success)
            if success:
                episode_landing_errors.append(landing_error)

            average_return = sum(episode_returns[-100:]) / len(
                episode_returns[-100:]
            )
            episode_metric = {
                "configuration": configuration,
                "episode": episode_counter,
                "step": step,
                "return": episode_return,
                "steps": current_episode_steps,
                "fuel_consumption": episode_fuel_consumption,
                "success": success,
                "landing_error": landing_error,
                "landing_speed": landing_speed,
                "final_horizontal_position": float(next_state[0]),
                "final_vertical_position": float(next_state[1]),
                "final_horizontal_velocity": float(next_state[2]),
                "final_vertical_velocity": float(next_state[3]),
                "epsilon": dqn.epsilon,
                "train_steps": dqn.train_steps,
                "average_return_last_100": average_return,
            }
            episode_metrics.append(episode_metric)

            with open(episode_metrics_path, "a") as metrics_file:
                metrics_file.write(json.dumps(episode_metric) + "\n")
                metrics_file.flush()

            print(
                f"Episode {episode_counter}: return = {episode_return}, "
                f"average = {average_return}, epsilon = {dqn.epsilon}",
                flush=True,
            )

            episode_counter += 1
            episode_return = 0
            current_episode_steps = 0
            episode_fuel_consumption = 0
            state, info = env.reset()

        if step % args.checkpoint_frequency == 0:
            checkpoint_path = f"{result_path}/checkpoint_{step}"
            os.makedirs(checkpoint_path, exist_ok=True)
            dqn.network.save(f"{checkpoint_path}/training_model.keras")
            dqn.target_network.save(f"{checkpoint_path}/target_network.keras")

            with open(f"{checkpoint_path}/training_state.pkl", "wb") as file:
                pkl.dump(
                    {
                        "experience_replay": dqn.experience_replay,
                        "epsilon": dqn.epsilon,
                        "train_steps": dqn.train_steps,
                        "step": step,
                        "episode_returns": episode_returns,
                        "episode_steps": episode_steps,
                        "current_episode_steps": current_episode_steps,
                        "episode_counter": episode_counter,
                        "episode_fuel_consumptions": episode_fuel_consumptions,
                        "episode_successes": episode_successes,
                        "episode_landing_errors": episode_landing_errors,
                        "episode_metrics": episode_metrics,
                        "configuration": configuration,
                    },
                    file,
                )

            with open(f"{checkpoint_path}/metrics.json", "w") as file:
                json.dump(
                    {
                        "configuration": configuration,
                        "epsilon": dqn.epsilon,
                        "train_steps": dqn.train_steps,
                        "step": step,
                        "episode_counter": episode_counter,
                        "episodes": episode_metrics,
                    },
                    file,
                    indent=4,
                )

    env.close()

    average_fuel_consumption = (
        sum(episode_fuel_consumptions) / len(episode_fuel_consumptions)
        if episode_fuel_consumptions
        else 0
    )
    success_rate = (
        sum(episode_successes) / len(episode_successes)
        if episode_successes
        else 0
    )
    average_landing_error = (
        sum(episode_landing_errors) / len(episode_landing_errors)
        if episode_landing_errors
        else None
    )

    dqn.network.save(f"{result_path}/training_model.keras")
    with open(f"{result_path}/final_metrics.json", "w") as file:
        json.dump(
            {
                "configuration": configuration,
                "episodes": episode_metrics,
                "episode_returns": episode_returns,
                "episode_steps": episode_steps,
                "final_epsilon": dqn.epsilon,
                "episode_fuel_consumptions": episode_fuel_consumptions,
                "average_fuel_consumption": average_fuel_consumption,
                "episode_successes": episode_successes,
                "success_rate": success_rate,
                "episode_landing_errors": episode_landing_errors,
                "average_landing_error": average_landing_error,
            },
            file,
            indent=4,
        )


if __name__ == "__main__":
    main()
