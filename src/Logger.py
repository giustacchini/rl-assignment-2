import json
import os
import pickle as pkl


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

    dqn.network.save(f"{checkpoint_location}/training_model.keras")

    dqn.target_network.save(f"{checkpoint_location}/target_network.keras")

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


def save_final_metrics(
    args,
    results_location,
    global_step,
    episode_returns,
    epsilon,
    episode_fuel_consumptions,
    average_fuel_consumption,
    episode_successes,
    episode_landing_errors,
    success_rate,
    average_landing_error,
):
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
            "target_update_frequency": args.target_update_frequency,
            "checkpoint_frequency": args.checkpoint_frequency,
        },
        "total_environment_steps": global_step,
        "episode_returns": episode_returns,
        "final_epsilon": epsilon,
        "episode_fuel_consumptions": episode_fuel_consumptions,
        "average_fuel_consumption": average_fuel_consumption,
        "episode_successes": episode_successes,
        "success_rate": success_rate,
        "episode_landing_errors": episode_landing_errors,
        "average_landing_error": average_landing_error,
    }

    with open(
        f"{results_location}/final_metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)
