# python
# lunar_env.py


import os
import json
import argparse
import keras
import gymnasium as gym
import pickle as pkl
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
    parser.add_argument("--checkpoint-frequency", type=int, default=10000)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main():
    # save model and metric here using model.save()  
    project_location = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(f"results/{project_location}", exist_ok=True)

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

    initial_step = 1

    state, info = env.reset()

    episode_returns = []
    episode_fuel_consumptions = []
    episode_successes = []
    episode_landing_errors = []

    episode_counter = 1
    episode_return = 0
    episode_fuel_consumption = 0

    if args.resume is not None:
        with open(f"{args.resume}/training_state.pkl", "rb") as f:
            loaded = pkl.load(f)
        initial_step = loaded["step"] + 1
        dqn.network = keras.saving.load_model(f"{args.resume}/training_model.keras")
        dqn.target_network = keras.saving.load_model(f"{args.resume}/target_network.keras")
        dqn.epsilon = loaded["epsilon"]
        dqn.experience_replay = loaded["experience_replay"]
        dqn.train_steps = loaded["train_steps"]
        episode_returns = loaded["episode_returns"]
        episode_counter = loaded["episode_counter"]
        episode_fuel_consumptions = loaded["episode_fuel_consumptions"]
        episode_successes = loaded["episode_successes"]
        episode_landing_errors = loaded["episode_landing_errors"]

    for step in range(initial_step, args.steps + 1):
        action = dqn.choose_action(state)
        current_state = state
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_done = terminated or truncated

        # Track fuel consumption for the current episode
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
            states, actions, rewards, next_states, dones = dqn.sample_experiences(batch_size=args.batch_size)
            dqn.train(states, actions, rewards, next_states, dones)
        state = next_state

        if episode_done:
            episode_returns.append(episode_return)
            episode_fuel_consumptions.append(episode_fuel_consumption)
            episode_successes.append(reward == 100)

            # Measure distance from the center for successful landings
            if reward == 100:
                episode_landing_errors.append(abs(next_state[0]))
            average_return = sum(episode_returns[-100:]) / len(episode_returns[-100:])
            print(
                f"Episode {episode_counter}: return = {episode_return}, average = {average_return}, epsilon = {dqn.epsilon}"
            )
            episode_counter += 1
            episode_return = 0
            episode_fuel_consumption = 0
            state, info = env.reset()

        if step % args.checkpoint_frequency == 0:
            temp_location = f"results/{project_location}/checkpoint_{step}"
            os.makedirs(temp_location, exist_ok=True)
            dqn.network.save(f"{temp_location}/training_model.keras")
            dqn.target_network.save(f"{temp_location}/target_network.keras")
            with open(f"{temp_location}/training_state.pkl", "wb") as f:
                pkl.dump({
                    "experience_replay": dqn.experience_replay,
                    "epsilon": dqn.epsilon,
                    "train_steps": dqn.train_steps,
                    "step": step,
                    "episode_returns": episode_returns,
                    "episode_counter": episode_counter,
                    "episode_fuel_consumptions": episode_fuel_consumptions,
                    "episode_successes": episode_successes,
                    "episode_landing_errors": episode_landing_errors
                }, f)
            metrics = {
                "epsilon": dqn.epsilon,
                "train_steps": dqn.train_steps,
                "step": step,
                "episode_return": episode_return,
                "episode_counter": episode_counter
            }
            with open(f"{temp_location}/metrics.json", "w") as f:
                f.write(json.dumps(metrics, indent=4))    

    env.close()

    # Calculate evaluation metrics for the whole training run
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
        "final_epsilon": dqn.epsilon,
        "episode_fuel_consumptions": episode_fuel_consumptions,
        "average_fuel_consumption": average_fuel_consumption,
        "episode_successes": episode_successes,
        "success_rate": success_rate,
        "episode_landing_errors": episode_landing_errors,
        "average_landing_error": average_landing_error,
    }
    with open(f"results/{project_location}/final_metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
