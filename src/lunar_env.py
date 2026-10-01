# lunar_env.py

import os
import json
import argparse
import pickle as pkl
from datetime import datetime

import keras
import gymnasium as gym

from dqn import DQN


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


def save_checkpoint(
    checkpoint_location,
    dqn,
    global_step,
    episode,
    episode_returns,
    episode_fuel_consumptions,
    episode_successes,
    episode_landing_errors,
):
    os.makedirs(checkpoint_location, exist_ok=True)

    dqn.network.save(
        f"{checkpoint_location}/training_model.keras"
    )

    dqn.target_network.save(
        f"{checkpoint_location}/target_network.keras"
    )

    training_state = {
        "experience_replay": dqn.experience_replay,
        "epsilon": dqn.epsilon,
        "train_steps": dqn.train_steps,
        "global_step": global_step,
        "episode": episode,
        "episode_returns": episode_returns,
        "episode_fuel_consumptions": episode_fuel_consumptions,
        "episode_successes": episode_successes,
        "episode_landing_errors": episode_landing_errors,
    }

    with open(
        f"{checkpoint_location}/training_state.pkl",
        "wb",
    ) as f:
        pkl.dump(training_state, f)

    checkpoint_metrics = {
        "global_step": global_step,
        "episode": episode,
        "epsilon": dqn.epsilon,
        "train_steps": dqn.train_steps,
        "completed_episodes": len(episode_returns),
    }

    with open(
        f"{checkpoint_location}/metrics.json",
        "w",
    ) as f:
        json.dump(checkpoint_metrics, f, indent=4)


def main():
    args = parse_args()

    if args.batch_size > args.replay_capacity:
        raise ValueError(
            "batch-size cannot be larger than replay-capacity"
        )

    project_location = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_location = f"results/{project_location}"

    os.makedirs(results_location, exist_ok=True)

    render_mode = "human" if args.render else None

    env = gym.make(
        "LunarLander-v3",
        render_mode=render_mode,
    )

    dqn = DQN(
        hidden_layers=args.hidden_layers,
        target_update_frequency=args.target_update_frequency,
        gamma=args.gamma,
        replay_capacity=args.replay_capacity,
        learning_rate=args.learning_rate,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
    )

    # Training history
    episode_returns = []
    episode_fuel_consumptions = []
    episode_successes = []
    episode_landing_errors = []

    # Global training progress
    global_step = 0
    initial_episode = 1

    # Resume checkpoint if requested
    if args.resume is not None:
        with open(
            f"{args.resume}/training_state.pkl",
            "rb",
        ) as f:
            loaded = pkl.load(f)

        dqn.network = keras.saving.load_model(
            f"{args.resume}/training_model.keras"
        )

        dqn.target_network = keras.saving.load_model(
            f"{args.resume}/target_network.keras"
        )

        dqn.epsilon = loaded["epsilon"]
        dqn.experience_replay = loaded["experience_replay"]
        dqn.train_steps = loaded["train_steps"]

        global_step = loaded["global_step"]

        episode_returns = loaded["episode_returns"]
        episode_fuel_consumptions = (
            loaded["episode_fuel_consumptions"]
        )
        episode_successes = loaded["episode_successes"]
        episode_landing_errors = (
            loaded["episode_landing_errors"]
        )

        # The checkpoint can occur in the middle of an episode.
        # We cannot restore the exact Gymnasium environment state,
        # therefore that episode starts again from env.reset().
        initial_episode = loaded["episode"]

    # -------------------------
    # Episode training loop
    # -------------------------

    for episode in range(
        initial_episode,
        args.episodes + 1,
    ):
        state, info = env.reset()

        episode_return = 0.0
        episode_fuel_consumption = 0.0

        episode_done = False

        while not episode_done:
            action = dqn.choose_action(state)

            current_state = state

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            episode_done = terminated or truncated
            global_step += 1

            # -------------------------
            # Fuel consumption
            # -------------------------

            if action == 2:
                # Main engine
                episode_fuel_consumption += 0.3

            elif action in (1, 3):
                # Orientation engines
                episode_fuel_consumption += 0.03

            episode_return += reward

            # -------------------------
            # Experience replay
            # -------------------------

            # Store EVERY transition.
            dqn.update_experience_replay(
                current_state,
                action,
                reward,
                next_state,
                terminated,
            )

            # Train once enough experiences exist.
            if len(dqn.experience_replay) >= args.batch_size:
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    terminateds,
                ) = dqn.sample_experiences(
                    batch_size=args.batch_size
                )

                dqn.train(
                    states,
                    actions,
                    rewards,
                    next_states,
                    terminateds,
                )

            state = next_state

            # -------------------------
            # Checkpoint every N steps
            # -------------------------

            if (
                args.checkpoint_frequency > 0
                and global_step % args.checkpoint_frequency == 0
            ):
                checkpoint_location = (
                    f"{results_location}/"
                    f"checkpoint_{global_step}"
                )

                save_checkpoint(
                    checkpoint_location,
                    dqn,
                    global_step,
                    episode,
                    episode_returns,
                    episode_fuel_consumptions,
                    episode_successes,
                    episode_landing_errors,
                )

        # -------------------------
        # Episode finished
        # -------------------------

        episode_returns.append(
            float(episode_return)
        )

        episode_fuel_consumptions.append(
            float(episode_fuel_consumption)
        )

        # Keep the existing success criterion for now.
        successful_landing = reward == 100
        episode_successes.append(successful_landing)

        if successful_landing:
            episode_landing_errors.append(
                float(abs(next_state[0]))
            )

        average_return = (
            sum(episode_returns[-100:])
            / len(episode_returns[-100:])
        )

        print(
            f"Episode {episode}: "
            f"steps = {global_step}, "
            f"return = {episode_return:.2f}, "
            f"average = {average_return:.2f}, "
            f"fuel = {episode_fuel_consumption:.2f}, "
            f"epsilon = {dqn.epsilon:.4f}"
        )

    env.close()

    # -------------------------
    # Final aggregate metrics
    # -------------------------

    average_fuel_consumption = (
        sum(episode_fuel_consumptions)
        / len(episode_fuel_consumptions)
        if episode_fuel_consumptions
        else 0
    )

    success_rate = (
        sum(episode_successes)
        / len(episode_successes)
        if episode_successes
        else 0
    )

    average_landing_error = (
        sum(episode_landing_errors)
        / len(episode_landing_errors)
        if episode_landing_errors
        else None
    )

    # -------------------------
    # Save final model
    # -------------------------

    dqn.network.save(
        f"{results_location}/training_model.keras"
    )

    dqn.target_network.save(
        f"{results_location}/target_network.keras"
    )

    # -------------------------
    # Save final metrics
    # -------------------------

    metrics = {
        "configuration": {
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "hidden_layers": args.hidden_layers,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "epsilon_decay": args.epsilon_decay,
            "epsilon_min": args.epsilon_min,
            "replay_capacity": args.replay_capacity,
            "target_update_frequency": (
                args.target_update_frequency
            ),
            "checkpoint_frequency": (
                args.checkpoint_frequency
            ),
        },

        "total_environment_steps": global_step,

        "episode_returns": episode_returns,
        "final_epsilon": dqn.epsilon,

        "episode_fuel_consumptions": (
            episode_fuel_consumptions
        ),
        "average_fuel_consumption": (
            average_fuel_consumption
        ),

        "episode_successes": episode_successes,
        "success_rate": success_rate,

        "episode_landing_errors": (
            episode_landing_errors
        ),
        "average_landing_error": (
            average_landing_error
        ),
    }

    with open(
        f"{results_location}/final_metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()